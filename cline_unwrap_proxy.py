#!/usr/bin/env python3
"""
Pass-through proxy that unwraps the cline.bot response format.

cline.bot returns responses wrapped as:
    {"data": {standard_openai_response}, "success": true}

LiteLLM's OpenAI parser expects the standard response at the top level.
This proxy forwards requests to cline.bot and unwraps the "data" field.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
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
            with urlopen(req, timeout=600) as resp:
                status = resp.status
                resp_body = resp.read()
        except Exception as e:
            # Handle HTTP errors from upstream
            status = getattr(e, "code", 500)
            resp_body = getattr(e, "read", lambda: b"")()
            if not resp_body:
                resp_body = str(e).encode()

        # Parse and unwrap the "data" field if present
        try:
            parsed = json.loads(resp_body)
            if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], dict):
                # Unwrap: use the inner data as the response
                resp_body = json.dumps(parsed["data"]).encode()
        except (json.JSONDecodeError, ValueError):
            pass  # Not JSON, pass through as-is

        # Send response back
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_POST(self):
        self._handle("POST")

    def do_GET(self):
        self._handle("GET")

    def log_message(self, format, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", "5001"))
    server = HTTPServer(("127.0.0.1", port), ProxyHandler)
    print(f"cline unwrap proxy listening on 127.0.0.1:{port}", flush=True)
    server.serve_forever()