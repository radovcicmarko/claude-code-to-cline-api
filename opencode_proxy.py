#!/usr/bin/env python3
"""
Routing proxy for OpenCode Go.

Claude Code sends Anthropic-format requests to this proxy. It routes:
  - Anthropic-native models (Qwen, MiniMax)  → OpenCode Go /v1/messages directly
  - OpenAI-compatible models (DeepSeek, Kimi, etc.) → LiteLLM for translation

Flow:
  Claude Code → routing proxy (:listen_port)
    ├─ Anthropic-native model → https://opencode.ai/zen/go/v1/messages  (direct)
    └─ Other models → http://127.0.0.1:litellm_port/v1/messages       (via LiteLLM)
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

OPENCODE_API_BASE = os.environ.get(
    "OPENCODE_API_BASE", "https://opencode.ai/zen/go/v1")
LITELLM_BASE = os.environ.get(
    "LITELLM_BASE", "http://127.0.0.1:4001")

# Models that support Anthropic /v1/messages natively on OpenCode Go.
ANTHROPIC_NATIVE_MODELS = {
    "minimax-m3", "minimax-m2.7", "minimax-m2.5",
    "qwen3.8-max", "qwen3.8-flash",
    "qwen3.7-max", "qwen3.7-plus",
    "qwen3.6-plus",
}

# Maps LiteLLM model_name → actual OpenCode model ID.
# This lets the proxy decide routing based on the real model, even though
# Claude Code sends LiteLLM-level model names in requests.
# Must match the model_list in litellm-config-opencode.yaml.
#
# Entries where the OpenCode model ID is in ANTHROPIC_NATIVE_MODELS will
# be routed directly to OpenCode; everything else goes through LiteLLM.
MODEL_MAP = {
    "claude-opus-5": "deepseek-v4-pro",
    "claude-sonnet-5": "deepseek-v4-flash",
    "claude-haiku-5": "kimi-k3",
    # Add custom mappings below:
}


class ProxyHandler(BaseHTTPRequestHandler):
    litellm_auth = ""

    def _resolve_model(self, model_name):
        """Map a LiteLLM model name to the real OpenCode model ID."""
        return MODEL_MAP.get(model_name, model_name)

    def _route(self, body_bytes):
        """Determine target for the request."""
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            body = {}
        model_name = body.get("model", "")
        real_model = self._resolve_model(model_name)

        if real_model in ANTHROPIC_NATIVE_MODELS:
            return "opencode", real_model
        else:
            return "litellm", model_name  # forward LiteLLM name as-is

    def _handle(self, method):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Save LiteLLM auth from incoming request (Claude Code sends it)
        ProxyHandler.litellm_auth = self.headers.get("Authorization", "")

        route, model_for_url = self._route(body)

        # Update the model in the body if needed (when routing directly to OpenCode)
        if route == "opencode":
            # Rewrite body with the real OpenCode model ID
            try:
                parsed = json.loads(body) if body else {}
                parsed["model"] = model_for_url
                body = json.dumps(parsed).encode()
            except json.JSONDecodeError:
                pass

            url = OPENCODE_API_BASE.rstrip("/") + "/v1/messages"
            auth_key = os.environ.get("OPENCODE_API_KEY", "")
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-opencode-session": os.environ.get(
                    "OPENCODE_SESSION", "claude-code-opencode-1"),
            }
            if auth_key:
                headers["Authorization"] = "Bearer " + auth_key
        else:
            # Route through LiteLLM — forward body unchanged
            url = LITELLM_BASE.rstrip("/") + "/v1/messages"
            headers = {
                "Authorization": ProxyHandler.litellm_auth,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            # Forward x-opencode-session from Claude Code so LiteLLM
            # can pass it through to OpenCode Go for routing/prompt caching.
            session = self.headers.get("x-opencode-session")
            if session:
                headers["x-opencode-session"] = session

        req = Request(url, data=body, headers=headers, method=method)
        try:
            resp = urlopen(req, timeout=600)
        except Exception as e:
            status = getattr(e, "code", 500)
            resp_body = getattr(e, "read", lambda: b"")()
            if not resp_body:
                resp_body = str(e).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self._safe_write(resp_body)
            return

        # Stream the response through as it arrives so SSE chunks reach the
        # client immediately instead of stalling until the upstream finishes.
        with resp:
            self.send_response(resp.status)
            self.send_header(
                "Content-Type",
                resp.headers.get("Content-Type", "application/json"))
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self._safe_write(chunk)
            except Exception as e:
                # Client hung up or upstream stalled mid-stream — the response
                # is already committed, so just drop the connection.
                sys.stderr.write("stream aborted: %r\n" % (e,))
                self.close_connection = True

    def _safe_write(self, data):
        try:
            self.wfile.write(data)
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def do_POST(self):
        try:
            self._handle("POST")
        except (ConnectionResetError, BrokenPipeError):
            # Client hung up before/while we read the request — nothing to do.
            sys.stderr.write("client disconnected mid-request\n")
            self.close_connection = True

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        body = json.dumps({
            "object": "list",
            "data": [
                {"id": m, "object": "model"}
                for m in sorted(set(MODEL_MAP.values()))
            ]
        }).encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._safe_write(body)

    def log_message(self, format, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def main():
    port = int(os.environ.get("PROXY_PORT", "4000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ProxyHandler)
    print("OpenCode routing proxy listening on 127.0.0.1:%d" % port, flush=True)
    print("  OpenCode API base: %s" % OPENCODE_API_BASE, flush=True)
    print("  LiteLLM base:      %s" % LITELLM_BASE, flush=True)
    an = [m for m in MODEL_MAP.values() if m in ANTHROPIC_NATIVE_MODELS]
    lit = [m for m in MODEL_MAP.values() if m not in ANTHROPIC_NATIVE_MODELS]
    if an:
        print("  -> direct to OpenCode: %s" % ", ".join(an), flush=True)
    if lit:
        print("  -> via LiteLLM:        %s" % ", ".join(lit), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()