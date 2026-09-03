#!/usr/bin/env python3
"""
LiteLLM control dashboard — supports both Cline and OpenCode Go backends.

Single-file, stdlib-only web dashboard that manages the proxy chain.

Cline backend:
    Claude Code -> LiteLLM (:litellm_port) -> unwrap proxy (:proxy_port) -> cline.bot

OpenCode Go backend:
    Claude Code -> routing proxy (:listen_port, default 4000)
        ├─ Anthropic-native model -> https://opencode.ai/zen/go/v1/messages  (direct)
        └─ OpenAI-compatible model -> LiteLLM (:4001) -> https://opencode.ai/zen/go/v1

Run:  python dashboard.py
Then open http://127.0.0.1:8080
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "litellm-config.yaml")
UNWRAP_SCRIPT = os.path.join(BASE_DIR, "cline_unwrap_proxy.py")
OPENCODE_PROXY_SCRIPT = os.path.join(BASE_DIR, "opencode_proxy.py")
LITELLM_OPENCODE_CONFIG = os.path.join(BASE_DIR, "litellm-config-opencode.yaml")

DASH_PORT = int(os.environ.get("DASH_PORT", "8080"))

CLINE_PASS_MODELS = [
    "glm-5.2", "kimi-k3", "kimi-k2.7-code", "kimi-k2.6",
    "deepseek-v4-pro", "deepseek-v4-flash", "mimo-v2.5", "mimo-v2.5-pro",
    "minimax-m3", "qwen3.8-max", "qwen3.7-max", "qwen3.7-plus",
]

# Zen free models (https://opencode.ai/docs/zen/) — served from a different
# base URL (https://opencode.ai/zen/v1) than the paid "go" endpoint.
ZEN_FREE_MODELS = [
    "big-pickle", "mimo-v2.5-free", "ling-3.0-flash-fin-free",
    "nemotron-3-ultra-free", "nemotron-3.5-lightning-free",
    "deepseek-v4-flash-free", "laguna-s-2.1-free",
]

OPENCODE_MODELS = {
    "/v1/chat/completions": [
        "glm-5.3-flash", "glm-5.3", "glm-5.2", "glm-5.1",
        "kimi-k3", "kimi-k2.7-code", "kimi-k2.6", "longcat-2.0",
        "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp",
        "mimo-v2.5", "mimo-v2.5-pro", "hy4-preview", "hy3",
        # Zen free models
        *ZEN_FREE_MODELS,
    ],
    "/v1/messages": [
        "minimax-m3", "minimax-m2.7", "minimax-m2.5",
        "qwen3.8-max", "qwen3.8-flash", "qwen3.7-max",
        "qwen3.7-plus", "qwen3.6-plus",
    ],
    "/v1/responses": [
        "grok-4.6", "gpt-5.6-luna", "muse-spark-1.3-contributor", "muse-spark-1.2-contributor",
        "muse-spark-1.3-contributor-free", "muse-spark-1.2-contributor-free",
    ],
}

# Models that can talk Anthropic natively → route directly
ANTHROPIC_NATIVE_MODELS = set(OPENCODE_MODELS["/v1/messages"])

# Flat list for dropdowns
ALL_OPENCODE_MODELS = []
for group in OPENCODE_MODELS.values():
    ALL_OPENCODE_MODELS.extend(group)

DEFAULT_PORTS = {"litellm_port": 4000, "proxy_port": 5001}
DEFAULT_OPENCODE_PORTS = {"litellm_port": 4001, "proxy_port": 4000}

STATE_PATH = os.path.join(BASE_DIR, "dashboard_state.json")


def _load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state):
    try:
        with open(STATE_PATH, "w", encoding="utf-8", newline="\n") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


def get_state_val(key, default):
    v = _load_state().get(key)
    return v if v is not None else default


def set_state_val(key, value):
    state = _load_state()
    state[key] = value
    _save_state(state)


def get_backend():
    return get_state_val("backend", "cline")


def set_backend(b):
    set_state_val("backend", b)


def get_litellm_port():
    return get_state_val("litellm_port", DEFAULT_PORTS["litellm_port"])


def set_litellm_port(port):
    set_state_val("litellm_port", int(port))


def get_oc_litellm_port():
    return get_state_val("oc_litellm_port", DEFAULT_OPENCODE_PORTS["litellm_port"])


def set_oc_litellm_port(port):
    set_state_val("oc_litellm_port", int(port))


def get_oc_proxy_port():
    return get_state_val("oc_proxy_port", DEFAULT_OPENCODE_PORTS["proxy_port"])


def set_oc_proxy_port(port):
    set_state_val("oc_proxy_port", int(port))


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------


def build_cline_config(*, opus_model, sonnet_model, cline_api_key, master_key,
                        drop_params, anthropic_route, use_chat_completions, proxy_port):
    def _b(v):
        return "true" if v else "false"

    blocks = []
    for model_name, model in (("claude-opus-5", opus_model), ("claude-sonnet-5", sonnet_model)):
        blocks.append(
            "  - model_name: " + model_name + "\n"
            "    litellm_params:\n"
            "      model: openai/cline-pass/" + model + "\n"
            "      api_base: http://127.0.0.1:" + str(proxy_port) + "\n"
            "      api_key: " + cline_api_key + "\n")

    settings = (
        "litellm_settings:\n"
        "  master_key: " + master_key + " # authentication for LiteLLM\n"
        "  drop_params: " + _b(drop_params) + "\n"
        "  anthropic_route: " + _b(anthropic_route) + " # enables the /v1/messages endpoint\n"
        "  use_chat_completions_url_for_anthropic_messages: "
        + _b(use_chat_completions) + " # route via /chat/completions, not /responses\n")

    text = "model_list:\n" + "\n".join(blocks) + "\n" + settings
    with open(CONFIG_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return text


def opencode_api_base(model):
    if model in ZEN_FREE_MODELS or model.endswith("-free"):
        return "https://opencode.ai/zen/v1"
    return "https://opencode.ai/zen/go/v1"


def build_opencode_litellm_config(*, opus_model, sonnet_model, api_key, master_key):
    blocks = []
    for model_name, model in (
        ("claude-opus-5", opus_model), ("claude-sonnet-5", sonnet_model)
    ):
        blocks.append(
            "  - model_name: " + model_name + "\n"
            "    litellm_params:\n"
            "      model: openai/" + model + "\n"
            "      api_base: " + opencode_api_base(model) + "\n"
            "      api_key: " + api_key + "\n")

    settings = (
        "litellm_settings:\n"
        "  master_key: " + master_key + " # authentication for LiteLLM\n"
        "  drop_params: true\n"
        "  anthropic_route: true\n"
        "  use_chat_completions_url_for_anthropic_messages: true\n")

    text = "model_list:\n" + "\n".join(blocks) + "\n" + settings
    with open(LITELLM_OPENCODE_CONFIG, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return text


def read_config():
    backend = get_backend()
    config_path = LITELLM_OPENCODE_CONFIG if backend == "opencode" else CONFIG_PATH

    if not os.path.exists(config_path):
        if backend == "opencode":
            build_opencode_litellm_config(
                opus_model=CLINE_PASS_MODELS[0],
                sonnet_model=CLINE_PASS_MODELS[5],
                api_key="YOUR_OPENCODE_API_KEY",
                master_key="sk-1234567890")
        else:
            build_cline_config(opus_model=CLINE_PASS_MODELS[0],
                                sonnet_model=CLINE_PASS_MODELS[5],
                                cline_api_key="YOUR_CLINE_API_KEY",
                                master_key="sk-1234567890",
                                drop_params=True, anthropic_route=True,
                                use_chat_completions=True,
                                proxy_port=DEFAULT_PORTS["proxy_port"])
    text = open(config_path, encoding="utf-8").read()

    model_by_name = {}
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if line.startswith("- model_name:"):
            name = line[len("- model_name:"):].strip()
            model_by_name[name] = {}

    current_name = None
    for raw in text.splitlines():
        line = raw.split("#")[0].rstrip()
        stripped = line.strip()
        if stripped.startswith("- model_name:"):
            current_name = stripped[len("- model_name:"):].strip()
            continue
        if current_name and line.startswith(" ") and ":" in stripped \
                and not stripped.startswith("model_list") and not stripped.startswith("litellm_settings"):
            k, v = stripped.split(":", 1)
            model_by_name[current_name][k.strip()] = v.strip()

    def _section_block(block_name):
        block, started = {}, False
        for raw in text.splitlines():
            line = raw.split("#")[0].rstrip()
            stripped = line.strip()
            if stripped == block_name + ":":
                started = True
                continue
            if started:
                if not stripped:
                    continue
                if not line.startswith(" "):
                    break
                if ":" in stripped and not stripped.startswith("- model_name:"):
                    k, v = stripped.split(":", 1)
                    block[k.strip()] = v.strip()
        return block

    settings_block = _section_block("litellm_settings")
    opus = model_by_name.get("claude-opus-5", {})
    sonnet = model_by_name.get("claude-sonnet-5", {})

    def _os_model(x):
        return x.replace("openai/", "").replace("cline-pass/", "")

    def _port_from_api_base(api_base, default):
        tail = api_base.rsplit(":", 1)
        if len(tail) == 2 and tail[1].isdigit():
            return int(tail[1])
        return default

    ports = dict(DEFAULT_PORTS)
    ports["litellm_port"] = get_litellm_port()
    for name, d in (("claude-opus-5", opus),):
        ports["proxy_port"] = _port_from_api_base(
            d.get("api_base", ""), ports["proxy_port"])

    cfg = {
        "opus_model": _os_model(opus.get("model", "")),
        "sonnet_model": _os_model(sonnet.get("model", "")),
        "cline_api_key": opus.get("api_key", ""),
        "opencode_api_key": opus.get("api_key", ""),
        "master_key": settings_block.get("master_key", "sk-1234567890"),
        "drop_params": _as_bool(settings_block.get("drop_params"), True),
        "anthropic_route": _as_bool(settings_block.get("anthropic_route"), True),
        "use_chat_completions": _as_bool(
            settings_block.get("use_chat_completions_url_for_anthropic_messages"), True),
        "ports": ports,
        "oc_ports": {
            "litellm_port": get_oc_litellm_port(),
            "proxy_port": get_oc_proxy_port(),
        },
        "backend": get_backend(),
    }
    if not cfg["opus_model"]:
        cfg["opus_model"] = CLINE_PASS_MODELS[0]
    if not cfg["sonnet_model"]:
        cfg["sonnet_model"] = CLINE_PASS_MODELS[5]
    return cfg


def _as_bool(v, default):
    if v is None:
        return default
    return str(v).strip().lower() == "true"


# ---------------------------------------------------------------------------
# Process manager
# ---------------------------------------------------------------------------

IS_WINDOWS = os.name == "nt"
LOG_LIMIT = 1500


def _no_window():
    return subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0


class ManagedProcess:
    def __init__(self, name):
        self.name = name
        self.proc = None
        self.log = deque(maxlen=LOG_LIMIT)
        self.lock = threading.Lock()

    @property
    def running(self):
        p = self.proc
        return p is not None and p.poll() is None

    def start(self, cmd, extra_env=None):
        with self.lock:
            if self.running:
                return False
            self.log.clear()
            env = os.environ.copy()
            if extra_env:
                env.update(extra_env)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=BASE_DIR, env=env,
                creationflags=_no_window(), encoding="utf-8", errors="replace",
                bufsize=1, universal_newlines=True,
                start_new_session=not IS_WINDOWS)
            self.proc = proc
        threading.Thread(target=self._tail, args=(proc,), daemon=True).start()
        return True

    def _tail(self, proc):
        try:
            for line in proc.stdout:
                self.log.append(line.rstrip("\n"))
        except ValueError:
            pass
        with self.lock:
            if self.proc is proc:
                self.proc = None

    def stop(self):
        with self.lock:
            p = self.proc
            self.proc = None
        if p is None:
            return
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True)
            else:
                os.killpg(os.getpgid(p.pid), 9)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def get_log(self, lines):
        return list(self.log)[-lines:]


# Shared processes
unwrap_proc = ManagedProcess("unwrap proxy")
litellm_proc = ManagedProcess("litellm")
opencode_proxy_proc = ManagedProcess("opencode proxy")


def start_chain(cfg):
    """Start the full proxy chain based on backend."""
    backend = cfg.get("backend", "cline")
    try:
        if backend == "opencode":
            oc_litellm_port = get_oc_litellm_port()
            oc_proxy_port = get_oc_proxy_port()
            if not opencode_proxy_proc.running:
                env = {"LITELLM_BASE": "http://127.0.0.1:" + str(oc_litellm_port)}
                opencode_proxy_proc.start([sys.executable, OPENCODE_PROXY_SCRIPT],
                                          extra_env={"PROXY_PORT": str(oc_proxy_port), **env})
            time.sleep(0.6)
            if not litellm_proc.running:
                litellm_proc.start(
                    ["litellm", "--config", LITELLM_OPENCODE_CONFIG,
                     "--port", str(oc_litellm_port)])
            return True, "started"
        else:
            # Cline: unwrap proxy → litellm
            if not unwrap_proc.running:
                unwrap_proc.start([sys.executable, UNWRAP_SCRIPT],
                                  extra_env={"PROXY_PORT": str(cfg["ports"]["proxy_port"])})
            time.sleep(0.6)
            if not litellm_proc.running:
                litellm_proc.start(
                    ["litellm", "--config", CONFIG_PATH,
                     "--port", str(cfg["ports"]["litellm_port"])])
            return True, "started"
    except Exception as e:
        return False, str(e)


def stop_chain():
    litellm_proc.stop()
    unwrap_proc.stop()
    opencode_proxy_proc.stop()
    return True


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class AppHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeClineDashboard/1.0"

    def _send(self, body, ctype, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj).encode("utf-8"),
                   "application/json; charset=utf-8", status)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _route(self, method):
        path = self.path.split("?", 1)[0]
        if method == "GET" and path in ("/", "/index.html"):
            self._send(render_html().encode("utf-8"), "text/html; charset=utf-8")
        elif method == "GET" and path == "/api/status":
            backend = get_backend()
            if backend == "opencode":
                self._json({
                    "opencode_proxy": {"state": "running" if opencode_proxy_proc.running else "stopped"},
                    "litellm": {"state": "running" if litellm_proc.running else "stopped"},
                    "backend": backend,
                })
            else:
                self._json({
                    "unwrap": {"state": "running" if unwrap_proc.running else "stopped"},
                    "litellm": {"state": "running" if litellm_proc.running else "stopped"},
                    "backend": backend,
                })
        elif method == "GET" and path == "/api/config":
            cfg = read_config()
            cfg["backend"] = get_backend()
            cfg["all_opencode_models"] = ALL_OPENCODE_MODELS
            cfg["opencode_groups"] = OPENCODE_MODELS
            cfg["anthropic_native_models"] = sorted(ANTHROPIC_NATIVE_MODELS)
            self._json(cfg)
        elif method == "GET" and path.startswith("/api/proxy/logs"):
            self._logs()
        elif method == "POST" and path == "/api/config":
            self._save()
        elif method == "POST" and path == "/api/proxy/start":
            ok, detail = start_chain(read_config())
            self._json({"ok": ok, "detail": detail})
        elif method == "POST" and path == "/api/proxy/stop":
            stop_chain()
            self._json({"ok": True})
        else:
            self._send(b"Not Found", "text/plain", 404)

    def _logs(self):
        qs = parse_qs(urlparse(self.path).query)
        which = qs.get("proc", ["litellm"])[0]
        if which == "opencode_proxy":
            proc = opencode_proxy_proc
        elif which == "unwrap":
            proc = unwrap_proc
        else:
            proc = litellm_proc
        try:
            lines = int(qs.get("lines", ["200"])[0])
        except ValueError:
            lines = 200
        self._json({"lines": proc.get_log(lines), "running": proc.running})

    def _save(self):
        data = self._body()
        errors, cfg = validate_and_save(data)
        if errors:
            self._json({"ok": False, "errors": errors})
            return
        was_running = (litellm_proc.running or unwrap_proc.running
                       or opencode_proxy_proc.running)
        detail = ("config saved & proxy restarted" if was_running
                  else "config saved (proxy not running)")
        if was_running:
            stop_chain()
            start_chain(cfg)
        self._json({"ok": True, "detail": detail})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def log_message(self, fmt, *args):
        pass


def validate_and_save(data):
    errors = []
    backend = data.get("backend", "cline")

    def _model(v, label):
        v = (v or "").strip()
        if not v:
            errors.append(label + " model is required")
        elif v.startswith("openai/") or v.startswith("cline-pass/"):
            errors.append(label + " should be the bare model id, no prefix")
        return v

    def _key(v, label):
        v = (v or "").strip()
        if not v:
            errors.append(label + " is required")
        elif v == "YOUR_CLINE_API_KEY" or v == "YOUR_OPENCODE_API_KEY":
            errors.append("Set a real API key (placeholder not allowed)")
        return v

    set_backend(backend)

    if backend == "opencode":
        opus = _model(data.get("opus_model"), "Opus")
        sonnet = _model(data.get("sonnet_model"), "Sonnet")
        key = _key(data.get("opencode_api_key"), "OpenCode API key")
        master = "sk-1234567890"
        oc_ports = dict(DEFAULT_OPENCODE_PORTS)
        try:
            oc_ports["litellm_port"] = int(data.get("oc_litellm_port") or DEFAULT_OPENCODE_PORTS["litellm_port"])
            oc_ports["proxy_port"] = int(data.get("oc_proxy_port") or DEFAULT_OPENCODE_PORTS["proxy_port"])
        except ValueError:
            errors.append("Ports must be whole numbers")
        if errors:
            return errors, None
        set_oc_litellm_port(oc_ports["litellm_port"])
        set_oc_proxy_port(oc_ports["proxy_port"])
        build_opencode_litellm_config(
            opus_model=opus, sonnet_model=sonnet, api_key=key, master_key=master)
        return [], {"backend": backend, "oc_ports": oc_ports}
    else:
        opus = _model(data.get("opus_model"), "Opus")
        sonnet = _model(data.get("sonnet_model"), "Sonnet")
        key = _key(data.get("cline_api_key"), "Cline API key")
        master = _key(data.get("master_key"), "LiteLLM master key")
        ports = dict(DEFAULT_PORTS)
        try:
            ports["litellm_port"] = int(data.get("litellm_port") or DEFAULT_PORTS["litellm_port"])
            ports["proxy_port"] = int(data.get("proxy_port") or DEFAULT_PORTS["proxy_port"])
        except ValueError:
            errors.append("Ports must be whole numbers")
        if errors:
            return errors, None
        set_litellm_port(ports["litellm_port"])
        build_cline_config(
            opus_model=opus, sonnet_model=sonnet, cline_api_key=key, master_key=master,
            drop_params=True, anthropic_route=True,
            use_chat_completions=True, proxy_port=ports["proxy_port"])
        return [], read_config()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_html():
    return _HTML_TEMPLATE.replace("@@CLINE_MODELS_JSON@@", json.dumps(CLINE_PASS_MODELS)) \
                         .replace("@@OPENCODE_MODELS_JSON@@", json.dumps(ALL_OPENCODE_MODELS)) \
                         .replace("@@OPENCODE_GROUPS_JSON@@", json.dumps(OPENCODE_MODELS))


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiteLLM Dashboard — Claude Code → Cline / OpenCode</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         background: #0d1117; color: #e6edf3; padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 2px; }
  .sub { color: #8b949e; font-size: 12px; margin-bottom: 8px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 18px 20px; margin-bottom: 16px; }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em;
             color: #8b949e; margin: 0 0 14px; }

  .row { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  .row label.f { min-width: 150px; color: #b1bac4; font-size: 12px; }
  select, input[type=text], input[type=password], input[type=number] {
    background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px;
    padding: 7px 9px; font-family: inherit; font-size: 13px; }
  select { min-width: 260px; }
  input[type=text], input[type=password] { flex: 1; min-width: 220px; max-width: 420px; }
  input[type=number] { width: 90px; }

  button { background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px;
           padding: 8px 16px; font-family: inherit; font-size: 13px; cursor: pointer; }
  button:hover { background: #30363d; }
  button.primary { background: #2ea043; border-color: #1f883d; }
  button.primary:hover { background: #3fb950; }
  button.stop { background: #da3633; border-color: #b62324; }
  button.stop:hover { background: #f85149; }
  button:disabled { opacity: .5; cursor: default; }
  button.small { padding: 4px 10px; font-size: 12px; }
  button.tog { min-width: 90px; text-align: center; }

  .status { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
            margin-right: 7px; vertical-align: middle; }
  .s-stopped  { background: #484f58; }
  .s-running  { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
  .s-starting { background: #d29922; box-shadow: 0 0 6px #d29922; }
  .badge { font-size: 13px; margin-right: 24px; }
  .proc-label { color: #b1bac4; margin-right: 6px; }

  #toast { position: fixed; top: 16px; right: 16px; padding: 10px 16px; border-radius: 6px;
           background: #161b22; border: 1px solid #30363d; color: #e6edf3; font-size: 13px;
           opacity: 0; transition: opacity .2s; max-width: 340px; z-index: 10; }
  #toast.show { opacity: 1; }
  #toast.err { border-color: #b62324; }

  .model-select-row { display: flex; gap: 8px; align-items: center; }
  .model-select-row input[type=text] { max-width: 260px; }

  .tabs { display: flex; gap: 4px; margin-bottom: 8px; flex-wrap: wrap; }
  .tabs button { padding: 5px 12px; }
  .tabs button.on { background: #30363d; border-color: #8b949e; }
  pre.log { background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
            padding: 12px; height: 280px; overflow: auto; margin: 0;
            font-size: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }

  .hint { background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
          padding: 12px 14px; font-size: 12px; color: #b1bac4; margin-top: 10px; }
  .hint code { color: #79c0ff; }
  .check { margin-right: 18px; color: #b1bac4; font-size: 13px; cursor: pointer; }
  .err { color: #f85149; font-size: 12px; margin: 0 0 10px; }
  .model-group-label { color: #8b949e; font-size: 11px; padding: 2px 8px; }
  optgroup { color: #e6edf3; background: #161b22; }
  hr { border: none; border-top: 1px solid #30363d; margin: 12px 0; }
</style>
</head>
<body>
  <h1>LiteLLM Dashboard</h1>
  <div class="sub">
    Claude Code → <span id="flow-label">backend</span>
  </div>

  <div id="toast"></div>

  <div class="card">
    <h2>Backend</h2>
    <div class="row">
      <button id="be-cline" class="tog" onclick="setBackend('cline')">Cline</button>
      <button id="be-opencode" class="tog" onclick="setBackend('opencode')">OpenCode Go</button>
      <span style="color:#8b949e;font-size:12px;margin-left:8px" id="backend-desc"></span>
    </div>
  </div>

  <div class="card">
    <h2>Proxy</h2>
    <div class="row">
      <span class="badge" id="status-badge1"></span>
      <span class="badge" id="status-badge2"></span>
      <button id="btn-start">Start proxy</button>
      <button id="btn-stop" class="stop">Stop proxy</button>
      <span id="busy" style="color:#8b949e;font-size:12px"></span>
    </div>
    <div class="hint" id="settings-hint"></div>
  </div>

  <div class="card" id="config-card">
    <h2 id="config-title">Models &amp; Keys</h2>

    <div class="row">
      <label class="f">Opus model</label>
      <span class="model-select-row">
        <select id="opus-select"></select>
        <input type="text" id="opus-custom" placeholder="custom model id" style="display:none">
      </span>
    </div>

    <div class="row">
      <label class="f">Sonnet model</label>
      <span class="model-select-row">
        <select id="sonnet-select"></select>
        <input type="text" id="sonnet-custom" placeholder="custom model id" style="display:none">
      </span>
    </div>

    <div class="row" id="key-row-cline">
      <label class="f">Cline API key</label>
      <input type="password" id="cline-key" placeholder="sk-...">
    </div>

    <div class="row" id="key-row-opencode" style="display:none">
      <label class="f">OpenCode API key</label>
      <input type="password" id="opencode-key" placeholder="your opencode key">
    </div>

    <div class="row" id="master-key-row">
      <label class="f">LiteLLM master key</label>
      <input type="password" id="master-key" placeholder="sk-1234567890">
    </div>
    <div id="cfg-errors" class="err"></div>

    <div class="row" style="margin-top:6px">
      <button id="btn-save" class="primary">Save configuration</button>
    </div>

    <div id="cline-ports">
      <div class="row">
        <label class="f">LiteLLM port</label>
        <input type="number" id="litellm-port" value="4000">
        <label class="f" style="margin-left:20px">unwrap proxy port</label>
        <input type="number" id="proxy-port" value="5001">
      </div>
    </div>

    <div id="opencode-ports" style="display:none">
      <div class="row">
        <label class="f">routing proxy port</label>
        <input type="number" id="oc-proxy-port" value="4000">
        <label class="f" style="margin-left:20px">LiteLLM port</label>
        <input type="number" id="oc-litellm-port" value="4001">
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Logs</h2>
    <div class="tabs" id="log-tabs">
      <button id="tab-litellm" class="on" onclick="setTab('litellm')">litellm</button>
      <button id="tab-unwrap" onclick="setTab('unwrap')">unwrap proxy</button>
      <button id="tab-opencode" style="display:none" onclick="setTab('opencode_proxy')">opencode proxy</button>
    </div>
    <pre class="log" id="logview">(no output yet)</pre>
  </div>

<script>
var CLINE_MODELS = @@CLINE_MODELS_JSON@@;
var OPENCODE_MODELS = @@OPENCODE_MODELS_JSON@@;
var OPENCODE_GROUPS = @@OPENCODE_GROUPS_JSON@@;
var state = { tab: "litellm", backend: "cline" };

// ---- model dropdowns ----
function buildSelect(selId, customId, value, modelList){
  var sel = document.getElementById(selId);
  sel.innerHTML = "";
  var found = false;

  // Group by opencode endpoint if using that list
  if (modelList === OPENCODE_MODELS && OPENCODE_GROUPS) {
    var groups = OPENCODE_GROUPS;
    var endpoints = Object.keys(groups);
    endpoints.forEach(function(ep){
      var grp = document.createElement("optgroup");
      grp.label = ep;
      groups[ep].forEach(function(m){
        var o = document.createElement("option");
        o.value = m; o.textContent = m;
        grp.appendChild(o);
        if (m === value) found = true;
      });
      sel.appendChild(grp);
    });
  } else {
    modelList.forEach(function(m){
      var o = document.createElement("option"); o.value = m; o.textContent = m;
      sel.appendChild(o);
      if (m === value) found = true;
    });
  }

  if (value && found) {
    sel.value = value;
    toggleCustom(selId, customId, value, true);
  } else {
    var label = value ? value + "  (custom)" : "Custom…";
    var o = document.createElement("option"); o.value = "__custom__"; o.textContent = label;
    sel.insertBefore(o, sel.firstChild);
    sel.value = "__custom__";
    toggleCustom(selId, customId, value, false);
  }
}

function toggleCustom(selId, customId, value, found){
  var custom = document.getElementById(customId);
  if (value && !found){
    custom.style.display = "";
    custom.value = value;
  } else {
    custom.style.display = "none";
  }
}
function modelValue(selId, customId){
  var sel = document.getElementById(selId);
  var custom = document.getElementById(customId);
  if (sel.value === "__custom__"){
    return custom.style.display === "none" ? "" : custom.value.trim();
  }
  return sel.value;
}
function onSelectChange(selId, customId){
  var sel = document.getElementById(selId);
  var custom = document.getElementById(customId);
  if (sel.value === "__custom__"){
    custom.style.display = "";
  } else {
    custom.style.display = "none";
  }
}

// ---- toast ----
function toast(msg, isErr){
  var t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "show" + (isErr ? " err" : "");
  clearTimeout(t._h);
  t._h = setTimeout(function(){ t.className = ""; }, 3500);
}

// ---- backend toggle ----
function setBackend(be){
  state.backend = be;
  ["cline","opencode"].forEach(function(b){
    var el = document.getElementById("be-" + b);
    el.style.background = b === be ? "#30363d" : "";
    el.style.borderColor = b === be ? "#8b949e" : "";
  });
  var desc = document.getElementById("backend-desc");
  if (be === "opencode") {
    desc.textContent = "Direct /v1/messages for Qwen/MiniMax, LiteLLM for others";
    document.getElementById("flow-label").textContent = "OpenCode Go";
    document.getElementById("key-row-cline").style.display = "none";
    document.getElementById("key-row-opencode").style.display = "";
    document.getElementById("master-key-row").style.display = "none";
    document.getElementById("cline-ports").style.display = "none";
    document.getElementById("opencode-ports").style.display = "";
    document.getElementById("tab-unwrap").style.display = "none";
    document.getElementById("tab-opencode").style.display = "";
    document.getElementById("config-title").textContent = "Models & OpenCode API Key";
  } else {
    desc.textContent = "Via unwrap proxy -> cline.bot";
    document.getElementById("flow-label").textContent = "Cline";
    document.getElementById("key-row-cline").style.display = "";
    document.getElementById("key-row-opencode").style.display = "none";
    document.getElementById("master-key-row").style.display = "";
    document.getElementById("cline-ports").style.display = "";
    document.getElementById("opencode-ports").style.display = "none";
    document.getElementById("tab-unwrap").style.display = "";
    document.getElementById("tab-opencode").style.display = "none";
    document.getElementById("config-title").textContent = "Models & Keys";
  }
  // reload model lists
  reloadModelLists(be);
  renderSettingsHint(be);
  refreshStatus();
}

function reloadModelLists(be){
  var models = be === "opencode" ? OPENCODE_MODELS : CLINE_MODELS;
  var el = document.getElementById("opus-select");
  el.onchange = function(){ onSelectChange("opus-select","opus-custom"); };
  var v = modelValue("opus-select","opus-custom");
  buildSelect("opus-select","opus-custom", v, models);
  el = document.getElementById("sonnet-select");
  el.onchange = function(){ onSelectChange("sonnet-select","sonnet-custom"); };
  v = modelValue("sonnet-select","sonnet-custom");
  buildSelect("sonnet-select","sonnet-custom", v, models);
}

function renderSettingsHint(be){
  var el = document.getElementById("settings-hint");
  var mk = (document.getElementById("master-key") || {}).value || "sk-1234567890";
  var port = (document.getElementById("litellm-port") || {}).value || "4000";
  if (be === "opencode") {
    var ocPort = (document.getElementById("oc-proxy-port") || {}).value || "4000";
    var snip = JSON.stringify(
      { env: { ANTHROPIC_BASE_URL: "http://localhost:" + ocPort,
               ANTHROPIC_AUTH_TOKEN: "sk-1234567890" },
        theme: "dark", model: "sonnet" }, null, 2);
    el.innerHTML = "Point Claude Code at the routing proxy by putting this in "
      + "<code>~/.claude/settings.json</code>:<br><br>"
      + "<pre>" + snip + "</pre><br>"
      + "Set <code>OPENCODE_API_KEY</code> env var for the dashboard process.";
  } else {
    var snip = JSON.stringify(
      { env: { ANTHROPIC_BASE_URL: "http://localhost:" + port,
               ANTHROPIC_AUTH_TOKEN: mk },
        theme: "dark", model: "sonnet" }, null, 2);
    el.innerHTML = "Point Claude Code at LiteLLM by putting this in "
      + "<code>~/.claude/settings.json</code>:<br><br>"
      + "<pre>" + snip + "</pre>";
  }
}

// ---- status / start / stop ----
function setStatusBadge(id, status, label){
  var el = document.getElementById(id);
  el.innerHTML = '<span class="status s-' + status + '"></span>'
    + '<span class="proc-label">' + label + '</span>'
    + status;
}
function refreshStatus(){
  fetch("/api/status").then(function(r){ return r.json(); })
  .then(function(d){
    if (d.backend === "opencode") {
      setStatusBadge("status-badge1", d.opencode_proxy.state, "routing proxy");
      setStatusBadge("status-badge2", d.litellm.state, "litellm");
    } else {
      setStatusBadge("status-badge1", d.unwrap.state, "unwrap proxy");
      setStatusBadge("status-badge2", d.litellm.state, "litellm");
    }
  }).catch(function(){});
}
function busy(t){ document.getElementById("busy").textContent = t || "";
  document.getElementById("btn-start").disabled = !!t;
  document.getElementById("btn-stop").disabled = !!t; }

function startProxy(){
  busy("starting…");
  refreshStatus();
  fetch("/api/proxy/start", {method:"POST"}).then(function(r){ return r.json(); })
    .then(function(d){ busy(""); toast(d.ok ? (d.detail||"started") : "start failed: " + d.detail, !d.ok); refreshStatus(); pollLogsNow(); })
    .catch(function(){ busy(""); toast("start request failed", true); });
}
function stopProxy(){
  busy("stopping…");
  refreshStatus();
  fetch("/api/proxy/stop", {method:"POST"}).then(function(r){ return r.json(); })
    .then(function(d){ busy(""); toast("stopped"); refreshStatus(); })
    .catch(function(){ busy(""); toast("stop request failed", true); });
}
document.getElementById("btn-start").addEventListener("click", startProxy);
document.getElementById("btn-stop").addEventListener("click", stopProxy);

// ---- save ----
function saveCfg(){
  var errors = document.getElementById("cfg-errors");
  errors.textContent = "";
  var payload = {
    backend: state.backend,
    opus_model: modelValue("opus-select","opus-custom"),
    sonnet_model: modelValue("sonnet-select","sonnet-custom"),
  };
  if (state.backend === "opencode") {
    payload.opencode_api_key = document.getElementById("opencode-key").value;
    payload.oc_litellm_port = document.getElementById("oc-litellm-port").value;
    payload.oc_proxy_port = document.getElementById("oc-proxy-port").value;
  } else {
    payload.cline_api_key = document.getElementById("cline-key").value;
    payload.master_key = document.getElementById("master-key").value;
    payload.litellm_port = document.getElementById("litellm-port").value;
    payload.proxy_port = document.getElementById("proxy-port").value;
  }
  fetch("/api/config", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)})
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (d.ok){
        toast(d.detail || "saved");
        loadCfg();
        pollLogsNow();
      } else {
        errors.textContent = d.errors.join("\\n");
        toast("validation failed", true);
      }
    })
    .catch(function(){ toast("save failed", true); });
}
document.getElementById("btn-save").addEventListener("click", saveCfg);

// ---- config load ----
function loadCfg(){
  fetch("/api/config").then(function(r){ return r.json(); }).then(function(d){
    document.getElementById("cline-key").value = d.cline_api_key || "";
    document.getElementById("opencode-key").value = d.opencode_api_key || "";
    document.getElementById("master-key").value = d.master_key || "sk-1234567890";
    document.getElementById("litellm-port").value = (d.ports && d.ports.litellm_port) || 4000;
    document.getElementById("proxy-port").value = (d.ports && d.ports.proxy_port) || 5001;
    document.getElementById("oc-litellm-port").value = (d.oc_ports && d.oc_ports.litellm_port) || 4001;
    document.getElementById("oc-proxy-port").value = (d.oc_ports && d.oc_ports.proxy_port) || 4000;
    var be = d.backend || "cline";
    setBackend(be);
    buildSelect("opus-select", "opus-custom", d.opus_model,
      be === "opencode" ? OPENCODE_MODELS : CLINE_MODELS);
    buildSelect("sonnet-select", "sonnet-custom", d.sonnet_model,
      be === "opencode" ? OPENCODE_MODELS : CLINE_MODELS);
  }).catch(function(){});
}

// ---- logs ----
function setTab(name){
  state.tab = name;
  document.querySelectorAll("#log-tabs button").forEach(function(b){
    b.className = b.id === "tab-" + name ? "on" : "";
  });
  var nm = name === "opencode_proxy" ? "opencode proxy" : (name === "unwrap" ? "unwrap proxy" : "litellm");
  document.getElementById("logview").dataset.label = nm;
  pollLogsNow();
}
function pollLogs(){
  if (document.hidden) { pollLogsNow(); return; }
  fetch("/api/proxy/logs?proc=" + state.tab + "&lines=400")
    .then(function(r){ return r.json(); }).then(function(d){
      var v = document.getElementById("logview");
      v.textContent = d.lines.join("\\n") || "(no output yet)";
      var atBottom = v.scrollTop + v.clientHeight >= v.scrollHeight - 40;
      if (atBottom) v.scrollTop = v.scrollHeight;
    }).catch(function(){});
}
function pollLogsNow(){
  refreshStatus();
  fetch("/api/proxy/logs?proc=" + state.tab + "&lines=400")
    .then(function(r){ return r.json(); }).then(function(d){
      var v = document.getElementById("logview");
      v.textContent = d.lines.join("\\n") || "(no output yet)";
      v.scrollTop = v.scrollHeight;
    }).catch(function(){});
}

["master-key","litellm-port","proxy-port","oc-proxy-port","oc-litellm-port"].forEach(function(id){
  var el = document.getElementById(id);
  if (el) el.addEventListener("input", function(){ renderSettingsHint(state.backend); });
});

setInterval(pollLogs, 2500);
setInterval(refreshStatus, 4000);
window.addEventListener("load", function(){ loadCfg(); setTab("litellm"); });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    read_config()
    try:
        server = HTTPServer(("127.0.0.1", DASH_PORT), AppHandler)
    except OSError as e:
        print("Dashboard: could not bind 127.0.0.1:%d — %s" % (DASH_PORT, e))
        sys.exit(1)
    print("LiteLLM dashboard running at http://127.0.0.1:%d" % DASH_PORT, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()