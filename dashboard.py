#!/usr/bin/env python3
"""
LiteLLM control dashboard.

Single-file, stdlib-only web dashboard that manages the LiteLLM proxy chain:

    Claude Code -> LiteLLM (:litellm_port) -> unwrap proxy (:proxy_port) -> cline.bot

It creates/maintains litellm-config.yaml, lets you edit API keys and the models
mapped to claude-opus and claude-sonnet from a browser, and starts/stops the
proxy chain. Saving a config while the chain is running restarts it.

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

DASH_PORT = int(os.environ.get("DASH_PORT", "8080"))

# ClinePass models from https://docs.cline.bot/getting-started/clinepass#models
CLINE_PASS_MODELS = [
    "glm-5.2", "kimi-k3", "kimi-k2.7-code", "kimi-k2.6",
    "deepseek-v4-pro", "deepseek-v4-flash", "mimo-v2.5", "mimo-v2.5-pro",
    "minimax-m3", "qwen3.8-max", "qwen3.7-max", "qwen3.7-plus",
]

DEFAULT_PORTS = {"litellm_port": 4000, "proxy_port": 5001}

# litellm-port lives here, not in litellm-config.yaml (which has no port entry).
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


def get_litellm_port():
    v = _load_state().get("litellm_port")
    return v if isinstance(v, int) and 0 < v < 65536 else DEFAULT_PORTS["litellm_port"]


def set_litellm_port(port):
    state = _load_state()
    state["litellm_port"] = int(port)
    _save_state(state)


# ---------------------------------------------------------------------------
# Config file helpers (fixed shape, no PyYAML dependency)
# ---------------------------------------------------------------------------


def build_config_text(*, opus_model, sonnet_model, cline_api_key, master_key,
                      drop_params, anthropic_route, use_chat_completions, proxy_port):
    """Regenerate litellm-config.yaml from editable settings."""
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


def read_config():
    """Parse litellm-config.yaml into a dict. Creates the default file if missing."""
    if not os.path.exists(CONFIG_PATH):
        build_config_text(opus_model=CLINE_PASS_MODELS[0],
                          sonnet_model=CLINE_PASS_MODELS[5],
                          cline_api_key="YOUR_CLINE_API_KEY",
                          master_key="sk-1234567890",
                          drop_params=True, anthropic_route=True,
                          use_chat_completions=True,
                          proxy_port=DEFAULT_PORTS["proxy_port"])
    text = open(CONFIG_PATH, encoding="utf-8").read()

    def _section_block(block_name):
        # Returns {key: value} parsed from indented `key: value` lines under block_name:
        # one for litellm_settings, one per model_list entry (keyed by model_name).
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
                    break  # back to top-level, block done
                if ":" in stripped and not stripped.startswith("- model_name:"):
                    k, v = stripped.split(":", 1)
                    block[k.strip()] = v.strip()
        return block

    model_by_name = {}
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if line.startswith("- model_name:"):
            name = line[len("- model_name:"):].strip()
            model_by_name[name] = {}

    # merge model_list params into the per-model dicts
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
        "master_key": settings_block.get("master_key", "sk-1234567890"),
        "drop_params": _as_bool(settings_block.get("drop_params"), True),
        "anthropic_route": _as_bool(settings_block.get("anthropic_route"), True),
        "use_chat_completions": _as_bool(
            settings_block.get("use_chat_completions_url_for_anthropic_messages"), True),
        "ports": ports,
    }
    # default models if the config had none
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
    """Run one child, tail its combined output, guard against double start."""

    def __init__(self, name):
        self.name = name
        self.proc = None
        self.log = deque(maxlen=LOG_LIMIT)
        self.lock = threading.Lock()

    @property
    def running(self):
        p = self.proc
        return p is not None and p.poll() is None

    def start(self, cmd, port_env=None, extra_env=None):
        with self.lock:
            if self.running:
                return False
            self.log.clear()
            env = os.environ.copy()
            if port_env:
                env["PROXY_PORT"] = str(port_env)
            if extra_env:
                env.update(extra_env)
            # Force UTF-8 I/O for the child: on Windows the default (cp1252)
            # encodes progress-spinner unicode and crashes litellm at startup.
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=BASE_DIR, env=env,
                creationflags=_no_window(), encoding="utf-8", errors="replace",
                bufsize=1, universal_newlines=True)
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


unwrap_proc = ManagedProcess("unwrap proxy")
litellm_proc = ManagedProcess("litellm")


def start_chain(cfg):
    """Start unwrap proxy then litellm. Returns (ok, detail)."""
    try:
        if not unwrap_proc.running:
            unwrap_proc.start([sys.executable, UNWRAP_SCRIPT],
                              port_env=cfg["ports"]["proxy_port"])
        time.sleep(0.6)  # let the proxy bind its socket first
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
    return True


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class AppHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeClineDashboard/1.0"

    # ---- helpers ----

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

    # ---- routes ----

    def _route(self, method):
        path = self.path.split("?", 1)[0]
        if method == "GET" and path in ("/", "/index.html"):
            self._send(render_html().encode("utf-8"), "text/html; charset=utf-8")
        elif method == "GET" and path == "/api/status":
            self._json({
                "unwrap": {"state": "running" if unwrap_proc.running else "stopped"},
                "litellm": {"state": "running" if litellm_proc.running else "stopped"},
            })
        elif method == "GET" and path == "/api/config":
            self._json(read_config())
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
        proc = litellm_proc if qs.get("proc", ["litellm"])[0] == "litellm" else unwrap_proc
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
        was_running = litellm_proc.running or unwrap_proc.running
        detail = ("config saved & proxy restarted" if was_running
                  else "config saved (proxy not running)")
        if was_running:
            stop_chain()
            start_chain(cfg)
        self._json({"ok": True, "detail": detail})

    # ---- verbs ----

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def log_message(self, fmt, *args):
        pass


def validate_and_save(data):
    """Validate POSTed config, write file on success. Returns (errors, config)."""
    errors = []

    def _model(v, label):
        v = (v or "").strip()
        if not v:
            errors.append(label + " model is required")
        elif v.startswith("openai/") or v.startswith("cline-pass/"):
            errors.append(label + " model should be the bare model id, no openai/cline-pass prefix")
        return v

    def _key(v, label):
        v = (v or "").strip()
        if not v:
            errors.append(label + " is required")
        elif v == "YOUR_CLINE_API_KEY":
            errors.append("Set a real Cline API key (placeholder not allowed)")
        return v

    def _b(v):
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "on", "yes")
        return bool(v)

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
    build_config_text(
        opus_model=opus, sonnet_model=sonnet, cline_api_key=key, master_key=master,
        drop_params=_b(data.get("drop_params")), anthropic_route=_b(data.get("anthropic_route")),
        use_chat_completions=_b(data.get("use_chat_completions")), proxy_port=ports["proxy_port"])
    return [], read_config()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_html():
    return _HTML_TEMPLATE.replace("@@MODELS_JSON@@", json.dumps(CLINE_PASS_MODELS))


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiteLLM Dashboard — Claude Code → Cline</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         background: #0d1117; color: #e6edf3; padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #8b949e; font-size: 12px; margin-bottom: 22px; }
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
  .model-select-row .custom-hint { color: #8b949e; font-size: 11px; }

  .tabs { display: flex; gap: 4px; margin-bottom: 8px; }
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
</style>
</head>
<body>
  <h1>LiteLLM Dashboard</h1>
  <div class="sub">Claude Code → LiteLLM → unwrap proxy → cline.bot</div>

  <div id="toast"></div>

  <div class="card">
    <h2>Proxy</h2>
    <div class="row">
      <span class="badge"><span class="status s-stopped" id="dot-unwrap"></span><span class="proc-label">unwrap proxy</span><span id="st-unwrap">stopped</span></span>
      <span class="badge"><span class="status s-stopped" id="dot-litellm"></span><span class="proc-label">litellm</span><span id="st-litellm">stopped</span></span>
      <button id="btn-start">Start proxy</button>
      <button id="btn-stop" class="stop">Stop proxy</button>
      <span id="busy" style="color:#8b949e;font-size:12px"></span>
    </div>
    <div class="hint">Point Claude Code at LiteLLM by putting this in <code>~/.claude/settings.json</code> (matches your master key / litellm port):</div>
    <pre class="hint" id="settings-snippet"></pre>
  </div>

  <div class="card">
    <h2>Models & Keys</h2>

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

    <div class="row">
      <label class="f">Cline API key</label>
      <input type="password" id="cline-key" placeholder="sk-...">
    </div>

    <div class="row">
      <label class="f">LiteLLM master key</label>
      <input type="password" id="master-key" placeholder="sk-1234567890">
    </div>
    <div id="cfg-errors" class="err"></div>

    <div class="row" style="margin-top:6px">
      <button id="btn-save" class="primary">Save configuration</button>
    </div>

    <div class="row">
      <label class="f">LiteLLM port</label>
      <input type="number" id="litellm-port" value="4000">
      <label class="f" style="margin-left:20px">unwrap proxy port</label>
      <input type="number" id="proxy-port" value="5001">
    </div>
  </div>

  <div class="card">
    <h2>Logs</h2>
    <div class="tabs">
      <button id="tab-litellm" class="on" onclick="setTab('litellm')">litellm</button>
      <button id="tab-unwrap" onclick="setTab('unwrap')">unwrap proxy</button>
    </div>
    <pre class="log" id="logview">(no output yet)</pre>
  </div>

<script>
// ---- state ----
var CANNED = @@MODELS_JSON@@;
var state = { tab: "litellm" };

// ---- model dropdowns ----
function buildSelect(selId, customId, value){
  var sel = document.getElementById(selId);
  sel.innerHTML = "";
  var add = function(v, label){
    var o = document.createElement("option");
    o.value = v; o.textContent = label; sel.appendChild(o);
  };
  var found = false;
  CANNED.forEach(function(m){
    add(m, m);
    if (m === value) found = true;
  });
  if (value && found) {
    sel.value = value;
    toggleCustom(selId, customId, value, true);
  } else {
    // value is empty, or a model not in the canned list (custom).
    var label = value ? value + "  (custom)" : "Custom…";
    add("__custom__", label);
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

// ---- toast ----
function toast(msg, isErr){
  var t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "show" + (isErr ? " err" : "");
  clearTimeout(t._h);
  t._h = setTimeout(function(){ t.className = ""; }, 3500);
}

// ---- status / start / stop ----
function setStatus(name, status){
  document.getElementById("st-" + name).textContent = status;
  document.getElementById("dot-" + name).className = "status s-" + status;
}
function refreshStatus(){
  fetch("/api/status").then(function(r){ return r.json(); }).then(function(d){
    ["unwrap","litellm"].forEach(function(n){
      setStatus(n, d[n].state);
    });
    renderSnippet();
  }).catch(function(){});
}
function renderSnippet(){
  var el = document.getElementById("settings-snippet");
  var mk = document.getElementById("master-key").value || "sk-1234567890";
  var port = document.getElementById("litellm-port").value || "4000";
  var snippet = {
    env: { ANTHROPIC_BASE_URL: "http://localhost:" + port,
           ANTHROPIC_AUTH_TOKEN: mk },
    theme: "dark", model: "sonnet"
  };
  el.textContent = "~/.claude/settings.json\\n" + JSON.stringify(snippet, null, 2);
}
["master-key","litellm-port"].forEach(function(id){
  document.getElementById(id).addEventListener("input", renderSnippet);
});

function busy(t){ document.getElementById("busy").textContent = t || ""; document.getElementById("btn-start").disabled = !!t; document.getElementById("btn-stop").disabled = !!t; }

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
    opus_model: modelValue("opus-select","opus-custom"),
    sonnet_model: modelValue("sonnet-select","sonnet-custom"),
    cline_api_key: document.getElementById("cline-key").value,
    master_key: document.getElementById("master-key").value,
    litellm_port: document.getElementById("litellm-port").value,
    proxy_port: document.getElementById("proxy-port").value
  };
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
    document.getElementById("cline-key").value = d.cline_api_key;
    document.getElementById("master-key").value = d.master_key;
    document.getElementById("litellm-port").value = (d.ports && d.ports.litellm_port) || 4000;
    document.getElementById("proxy-port").value = (d.ports && d.ports.proxy_port) || 5001;
    buildSelect("opus-select", "opus-custom", d.opus_model);
    buildSelect("sonnet-select", "sonnet-custom", d.sonnet_model);
    renderSnippet();
  }).catch(function(){});
}

// ---- logs ----
function setTab(name){
  state.tab = name;
  document.getElementById("tab-litellm").className = name==="litellm" ? "on" : "";
  document.getElementById("tab-unwrap").className = name==="unwrap" ? "on" : "";
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
      var running = d.running;
      if (state.tab === "litellm") setStatus("litellm", running ? "running":"stopped");
      else setStatus("unwrap", running ? "running":"stopped");
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
    # Ensure config exists so the first start has something to run.
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