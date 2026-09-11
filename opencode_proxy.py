#!/usr/bin/env python3
"""
Routing proxy for OpenCode Go.

Claude Code sends Anthropic-format requests to this proxy.
The proxy forwards requests to LiteLLM, which handles model mapping
and auth to OpenCode Go. It maps X-Claude-Code-Session-Id → x-opencode-session.

Flow:
  Claude Code → proxy (:listen_port) → LiteLLM → OpenCode Go /v1/messages

Model mapping is handled entirely by LiteLLM's config (litellm-config-opencode.yaml).
The proxy passes through model names unchanged to avoid double-mapping.
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


class ProxyHandler(BaseHTTPRequestHandler):
    def _route(self, body_bytes):
        """Determine target for the request."""
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            body = {}
        model_name = body.get("model", "")
        return "litellm", model_name

    def _handle(self, method):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Debug: log incoming headers from Claude Code
        sys.stderr.write("<<< INCOMING HEADERS: %s\n" % (
            json.dumps(dict(self.headers)),))
        sys.stderr.flush()

        route, model_name = self._route(body)

        url = LITELLM_BASE.rstrip("/") + "/v1/messages"
        headers = {
            "Authorization": self.headers.get("Authorization", ""),
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        # Map X-Claude-Code-Session-Id → x-opencode-session
        session_id = self.headers.get("X-Claude-Code-Session-Id", "")
        if session_id:
            headers["x-opencode-session"] = session_id
        # Forward all x- headers from the original request
        for key, val in self.headers.items():
            low = key.lower()
            if low.startswith("x-") and low != "x-opencode-session":
                headers[key] = val

        sys.stderr.write(">>> ROUTE=%s URL=%s HEADERS=%s\n" % (
            route, url, json.dumps(dict(headers)),))
        sys.stderr.flush()
        req = Request(url, data=body, headers={}, method=method)
        # Set headers directly on the Message object to preserve exact casing
        # (Request.add_header() title-cases keys, breaking x-opencode-session)
        for k, v in headers.items():
            req.headers[k] = v
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
                for m in ["claude-opus-5", "claude-sonnet-5", "claude-haiku-5"]
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
    print("  Model mapping: handled by LiteLLM config", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()