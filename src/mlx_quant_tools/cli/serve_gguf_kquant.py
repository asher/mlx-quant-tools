"""OpenAI-compatible HTTP server for GGUF K-quant models on MLX.

Loads a GGUF K-quant file via load_kquant_model (from gguf_runtime),
then serves /v1/chat/completions, /v1/completions, /v1/models, and /health.
Single-model, single-request-at-a-time, local-only.

Usage
-----
  mqt-serve-gguf /path/to/model.gguf
  mqt-serve-gguf /path/to/model.gguf --port 8080 --temp 0.7

  # curl test:
  curl http://127.0.0.1:8080/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step, generation_stream, wired_limit
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from mlx_quant_tools.gguf_runtime import load_kquant_model

log = logging.getLogger("serve-gguf-kquant")


# ---------------------------------------------------------------------------
# Vendored from mlx_lm.utils (private API, avoids breakage on version bumps)
# ---------------------------------------------------------------------------


def _parse_size(x: str) -> int:
    sizes = {"M": 1e6, "G": 1e9, "MB": 1e6, "GB": 1e9, "": 1}
    split = 0
    for xi in x:
        if not (xi.isdigit() or xi == "."):
            break
        split += 1
    digits = float(x[:split])
    size = (x[split:]).strip().upper()
    return int(digits * sizes[size])


# ---------------------------------------------------------------------------
# Stop-word checking
# ---------------------------------------------------------------------------


def _check_stop_words(text: str, stop_words: list[str]) -> tuple[str, bool]:
    for sw in stop_words:
        idx = text.find(sw)
        if idx >= 0:
            return text[:idx], True
    return text, False


# ---------------------------------------------------------------------------
# OpenAI response builders
# ---------------------------------------------------------------------------


def _make_chunk(
    request_id: str,
    model_name: str,
    delta_text: str | None,
    finish_reason: str | None,
    is_chat: bool,
) -> dict:
    choice: dict = {"index": 0, "finish_reason": finish_reason}
    if is_chat:
        delta = {"role": "assistant"}
        if delta_text:
            delta["content"] = delta_text
        choice["delta"] = delta
    else:
        choice["text"] = delta_text or ""
    return {
        "id": request_id,
        "object": "chat.completion.chunk" if is_chat else "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [choice],
    }


def _make_response(
    request_id: str,
    model_name: str,
    text: str,
    finish_reason: str,
    is_chat: bool,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    choice: dict = {"index": 0, "finish_reason": finish_reason}
    if is_chat:
        choice["message"] = {"role": "assistant", "content": text}
    else:
        choice["text"] = text
    return {
        "id": request_id,
        "object": "chat.completion" if is_chat else "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class KQuantHandler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None
    model_name = ""
    model_key = ""
    default_max_tokens = 4096
    default_temp = 0.0
    max_kv_size: int | None = None
    prompt_cache_store: LRUPromptCache | None = None

    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    # -- helpers --------------------------------------------------------

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _error(self, code: int, msg: str):
        self._send_json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

    # -- routes ---------------------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_name,
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._error(404, "not found")

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat()
        elif self.path == "/v1/completions":
            self._handle_text()
        else:
            self._error(404, "not found")

    # -- chat completions -----------------------------------------------

    def _handle_chat(self):
        body = self._read_body()
        messages = body.get("messages")
        if not messages:
            self._error(400, "messages is required")
            return

        tokenizer = self.tokenizer
        if tokenizer.chat_template is None:
            self._error(400, "model has no chat template")
            return

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            self._error(400, f"chat template error: {e}")
            return

        self._handle_generate(prompt, body, is_chat=True)

    # -- text completions -----------------------------------------------

    def _handle_text(self):
        body = self._read_body()
        prompt = body.get("prompt", "")
        if not prompt:
            self._error(400, "prompt is required")
            return
        self._handle_generate(prompt, body, is_chat=False)

    # -- generation core ------------------------------------------------

    def _handle_generate(self, prompt: str, body: dict, *, is_chat: bool):
        stream = body.get("stream", False)
        max_tokens = (
            body.get("max_tokens") or body.get("max_completion_tokens") or self.default_max_tokens
        )
        temp = body.get("temperature", self.default_temp)
        top_p = body.get("top_p", 1.0)

        stop = body.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]

        request_id = (
            f"chatcmpl-{uuid.uuid4().hex[:12]}" if is_chat else f"cmpl-{uuid.uuid4().hex[:12]}"
        )

        sampler = make_sampler(temp, top_p=top_p)

        tokenizer = self.tokenizer
        model = self.model

        add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=add_special)
        prompt_arr = mx.array(prompt_tokens)

        try:
            self._generate(
                model,
                tokenizer,
                prompt_arr,
                prompt_tokens,
                sampler=sampler,
                max_tokens=max_tokens,
                stop_words=stop,
                stream=stream,
                is_chat=is_chat,
                request_id=request_id,
            )
        except BrokenPipeError:
            log.warning("client disconnected during generation")
        except Exception:
            log.exception("generation error")
            try:
                self._error(500, "internal error")
            except Exception:
                pass

    def _generate(
        self,
        model,
        tokenizer,
        prompt_arr,
        prompt_tokens,
        *,
        sampler,
        max_tokens,
        stop_words,
        stream,
        is_chat,
        request_id,
    ):
        cache_store = self.prompt_cache_store
        model_key = self.model_key

        max_kv = self.max_kv_size

        # Fetch nearest cached KV state or create fresh
        if cache_store is not None:
            cached, remaining = cache_store.fetch_nearest_cache(model_key, prompt_tokens)
            if cached is not None:
                remaining_arr = mx.array(remaining) if remaining else prompt_arr[-1:]
                cache_hit_len = len(prompt_tokens) - len(remaining)
                log.info(
                    "cache hit: %d / %d prompt tokens cached", cache_hit_len, len(prompt_tokens)
                )
            else:
                cached = make_prompt_cache(model, max_kv_size=max_kv)
                remaining_arr = prompt_arr
        else:
            cached = make_prompt_cache(model, max_kv_size=max_kv)
            remaining_arr = prompt_arr

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        eos_ids = tokenizer.eos_token_ids

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._cors_headers()
            self.end_headers()

        accumulated = ""
        completion_tokens = 0
        finish_reason = None

        with wired_limit(model, [generation_stream]):
            for token, _logprobs in generate_step(
                remaining_arr,
                model,
                max_tokens=max_tokens,
                max_kv_size=max_kv,
                sampler=sampler,
                prompt_cache=cached,
            ):
                if token in eos_ids:
                    finish_reason = "stop"
                    break

                detokenizer.add_token(token)
                segment = detokenizer.last_segment
                accumulated += segment
                completion_tokens += 1

                if stop_words:
                    truncated, matched = _check_stop_words(accumulated, stop_words)
                    if matched:
                        accumulated = truncated
                        finish_reason = "stop"
                        break

                if stream and segment:
                    chunk = _make_chunk(request_id, self.model_name, segment, None, is_chat)
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()

            if finish_reason is None:
                finish_reason = "length"

            detokenizer.finalize()
            tail = detokenizer.last_segment
            if tail:
                accumulated += tail

        # Insert cache for future prefix reuse
        if cache_store is not None:
            cache_store.insert_cache(model_key, prompt_tokens, cached)

        if stream:
            final = _make_chunk(
                request_id, self.model_name, tail if tail else None, finish_reason, is_chat
            )
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            resp = _make_response(
                request_id,
                self.model_name,
                accumulated,
                finish_reason,
                is_chat,
                len(prompt_tokens),
                completion_tokens,
            )
            self._send_json(200, resp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="OpenAI-compatible server for GGUF K-quant models on MLX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("gguf", help="Path to GGUF file")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Default max completion tokens when not in request (default: 4096)",
    )
    ap.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default temperature when not in request (default: 0.0)",
    )
    ap.add_argument("--arch", help="Override architecture detection")
    ap.add_argument(
        "--target-prefix", default="", help="Prepend prefix to all remapped tensor names"
    )
    ap.add_argument(
        "--no-remap", action="store_true", help="Skip GGUF→HF name remap (use raw GGUF names)"
    )
    ap.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Max KV cache sequence length per request (default: unlimited)",
    )
    ap.add_argument(
        "--prompt-cache-size",
        type=int,
        default=10,
        help="Max number of KV caches to keep for prompt prefix reuse (default: 10)",
    )
    ap.add_argument(
        "--prompt-cache-bytes",
        type=_parse_size,
        default=None,
        help="Max total bytes for KV cache pool, e.g. '4G' (default: unlimited)",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    print(f"Loading {args.gguf} ...")
    model, config, tokenizer = load_kquant_model(
        args.gguf,
        arch=args.arch,
        target_prefix=args.target_prefix,
        no_remap=args.no_remap,
    )

    model_name = Path(args.gguf).stem
    model_key = args.gguf

    cache_max_bytes = args.prompt_cache_bytes if args.prompt_cache_bytes else 1 << 63
    prompt_cache_store = LRUPromptCache(
        max_size=args.prompt_cache_size,
        max_bytes=cache_max_bytes,
    )

    KQuantHandler.model = model
    KQuantHandler.tokenizer = tokenizer
    KQuantHandler.model_name = model_name
    KQuantHandler.model_key = model_key
    KQuantHandler.default_max_tokens = args.max_tokens
    KQuantHandler.default_temp = args.temp
    KQuantHandler.max_kv_size = args.max_kv_size
    KQuantHandler.prompt_cache_store = prompt_cache_store

    httpd = HTTPServer((args.host, args.port), KQuantHandler)
    print(f"\nServing {model_name} at http://{args.host}:{args.port}/v1")
    print("Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /health")
    print("Press Ctrl+C to stop.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
