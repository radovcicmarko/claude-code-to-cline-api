#!/usr/bin/env python3
"""
Pass-through proxy that unwraps the cline.bot response format.

cline.bot returns responses wrapped as:
    {"data": {standard_openai_response}, "success": true}

LiteLLM's OpenAI parser expects the standard response at the top level.
This proxy forwards requests to cline.bot and unwraps the "data" field.
SSE streams are relayed event-by-event as they arrive, unwrapping each one.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

CLINE_API_BASE = os.environ.get("CLINE_API_BASE", "https://api.cline.bot/api/v1")

class ProxyHandler(BaseHTTPRequestHandler):
    def _handle(self, method):
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Build the upstream URL (append path to api base)
        path = self.path
        if path.startswith("/chat/completions"):
            url = f"{CLINE_API_BASE}/chat/completions"
        elif path.startswith("/"):
            url = f"{CLINE_API_BASE}{path}"
        else:
            url = f"{CLINE_API_BASE}/{path}"

        # Read the Authorization header that LiteLLM sent us.
        # This comes from `api_key` in litellm-config.yaml, so the key
        # lives in one place and no CLINE_API_KEY env var is needed.
        auth_header = self.headers.get("Authorization", "")

        # Forward request to cline.bot, passing the auth through
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }
        req = Request(url, data=body, headers=headers, method=method)
        try:
            resp = urlopen(req, timeout=600)
        except Exception as e:
            # Handle HTTP errors from upstream
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

        if "text/event-stream" in resp.headers.get("Content-Type", ""):
            # A stream isn't valid JSON as a whole, so unwrap per event.
            self._stream_response(resp)
            return

        with resp:
            resp_body = resp.read()

        # Parse and unwrap the "data" field if present
        unwrapped = self._unwrap_json(resp_body)
        if unwrapped is not None:
            resp_body = unwrapped

        # Send response back
        self.send_response(resp.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self._safe_write(resp_body)

    def _stream_response(self, resp):
        # Stream the response through as it arrives so SSE chunks reach the
        # client immediately instead of stalling until the upstream finishes.
        with resp:
            self.send_response(resp.status)
            self.send_header(
                "Content-Type",
                resp.headers.get("Content-Type", "text/event-stream"))
            self.end_headers()
            buf = b""
            try:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n\n" in buf:
                        event, buf = buf.split(b"\n\n", 1)
                        self._safe_write(self._unwrap_sse_event(event) + b"\n\n")
                if buf.strip():
                    self._safe_write(self._unwrap_sse_event(buf) + b"\n\n")
            except Exception as e:
                # Client hung up or upstream stalled mid-stream — the response
                # is already committed, so just drop the connection.
                sys.stderr.write("stream aborted: %r\n" % (e,))
                self.close_connection = True

    # Returns the unwrapped payload if this is the cline envelope, else None
    def _unwrap_json(self, payload):
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return None
        if (isinstance(parsed, dict) and "data" in parsed
                and isinstance(parsed["data"], dict)):
            return json.dumps(parsed["data"]).encode()
        return None

    def _unwrap_sse_event(self, event):
        event = event.replace(b"\r\n", b"\n")
        out = []
        for line in event.split(b"\n"):
            if line.startswith(b"data:"):
                payload = line[5:].strip()
                unwrapped = self._unwrap_json(payload)
                if unwrapped is not None:
                    payload = unwrapped
                line = b"data: " + payload
            out.append(line)
        return b"\n".join(out)

    def _safe_write(self, data):
        try:
            self.wfile.write(data)
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _safe_handle(self, method):
        try:
            self._handle(method)
        except (ConnectionResetError, BrokenPipeError):
            # Client hung up before/while we read the request — nothing to do.
            sys.stderr.write("client disconnected mid-request\n")
            self.close_connection = True

    def do_POST(self):
        self._safe_handle("POST")

    def do_GET(self):
        self._safe_handle("GET")

    def log_message(self, format, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", "5001"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ProxyHandler)
    print(f"cline unwrap proxy listening on 127.0.0.1:{port}", flush=True)
    server.serve_forever()