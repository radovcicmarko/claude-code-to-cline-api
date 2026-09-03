# Claude Code → Cline API / OpenCode Go via LiteLLM

This project connects **Claude Code** (Anthropic's CLI coding agent) to either the
**Cline API** (api.cline.bot) or **OpenCode Go** (opencode.ai) using **LiteLLM** as
a translation proxy. It lets you use non-Anthropic models (DeepSeek, Qwen, Kimi, etc.)
from within Claude Code.

## Backends

### Cline

```
Claude Code → LiteLLM (:4000) → unwrap proxy (:5001) → cline.bot
```

1. Claude Code sends an **Anthropic**-format request to `POST /v1/messages` on LiteLLM.
2. LiteLLM translates it to the **OpenAI** format and forwards it as `POST /chat/completions`.
3. The **unwrap proxy** (`cline_unwrap_proxy.py`) forwards it to cline.bot and unwraps the response.
4. cline.bot returns the completion, which flows back up to Claude Code.

### OpenCode Go

```
Claude Code → routing proxy (:4000)
    ├─ Anthropic-native model (Qwen, MiniMax) → opencode.ai/zen/go/v1/messages  (direct)
    └─ Other models (DeepSeek, Kimi, GLM) → LiteLLM (:4001) → opencode.ai/zen/go/v1
```

1. Claude Code sends an **Anthropic**-format request to the routing proxy on port 4000.
2. The routing proxy checks the model:
   - **Anthropic-native models** (Qwen, MiniMax) → proxies directly to OpenCode Go's `/v1/messages` endpoint. No translation needed — Claude Code already speaks Anthropic.
   - **OpenAI-compatible models** (DeepSeek, Kimi, GLM, etc.) → forwards to LiteLLM on port 4001, which translates to OpenAI format and sends to OpenCode Go's `/v1/chat/completions`.
3. OpenCode Go returns standard-format responses — no unwrapping needed.
4. **Responses API models** (Grok, GPT-5.6-luna, Muse Spark) are **not supported** — see note below.

---

## Why is `cline_unwrap_proxy.py` needed?

There are **two** reasons a direct LiteLLM → cline.bot connection fails:

### 1. cline.bot wraps its response in a `data` field

cline.bot does **not** return a standard OpenAI `chat.completion` response. Instead, it wraps the real response inside a `data` object and adds a `success` flag:

```json
{
  "data": {
    "choices": [
      {
        "index": 0,
        "message": { "role": "assistant", "content": "..." },
        "finish_reason": "stop"
      }
    ],
    "id": "gen_...",
    "model": "deepseek/deepseek-v4-flash",
    "object": "chat.completion",
    "usage": { "prompt_tokens": 5, "completion_tokens": 8 }
  },
  "success": true
}
```

LiteLLM's OpenAI parser expects `choices` to be at the **top level** of the response. Because the actual data is nested inside `data.choices`, LiteLLM fails with:

```
litellm.APIError: provider returned a response with no 'choices'. Raw keys: ['id', 'choices', ..., 'data', 'success']
```

**The unwrap proxy solves this** by parsing the response and promoting the `data` object to the top level, so LiteLLM receives a standard response it can parse:

```json
{
  "choices": [ ... ],
  "id": "gen_...",
  "model": "deepseek/deepseek-v4-flash",
  "object": "chat.completion",
  "usage": { ... }
}
```

### 2. LiteLLM routes Anthropic requests to the wrong endpoint

By default, LiteLLM v1.95+ routes Anthropic `/v1/messages` requests to the **OpenAI Responses API** (`/responses` endpoint) instead of the **Chat Completions API** (`/chat/completions`). This happens because the deployment uses `openai` as its provider.

The cline.bot API only supports `/chat/completions`, so it returns `404 Not Found`, which puts the deployment into **cooldown**, eventually causing:

```
litellm.types.router.RouterRateLimitError: No deployments available for selected model
```

This is fixed in the LiteLLM config with:

```yaml
litellm_settings:
  use_chat_completions_url_for_anthropic_messages: true
```

This forces LiteLLM to use `/chat/completions` for Anthropic requests instead of `/responses`.

---

## OpenCode Go endpoint types

OpenCode Go serves models from three different endpoints. The routing proxy handles
this automatically, but here's how they map:

| Endpoint | Protocol | Models | How it's routed |
|---|---|---|---|
| `/v1/chat/completions` | OpenAI Chat | DeepSeek V4, Kimi K2-3, GLM 5.x, MiMo V2, Hy3/4, LongCat 2.0 | Via LiteLLM (Anthropic → OpenAI translation) |
| `/v1/messages` | Anthropic Messages | Qwen 3.6-3.8, MiniMax M2.5-M3 | Direct to OpenCode (no translation needed) |
| `/v1/responses` | OpenAI Responses | Grok 4.6, GPT-5.6-luna, Muse Spark | **Not supported** — neither Anthropic nor Chat format maps to Responses API |

## Prerequisites

- **Python 3.7+**
- **LiteLLM** installed (see [How to Run](#how-to-run))
- A **Cline API key** from [cline.bot](https://cline.bot) OR an **OpenCode Go subscription** from [opencode.ai](https://opencode.ai/auth) ($10/month)
- **Claude Code** installed

---

## Known dependency issue: FastAPI version conflict

LiteLLM currently has a dependency conflict with newer versions of FastAPI. If you see this error when starting LiteLLM:

```
ImportError: cannot import name 'get_flat_dependant' from 'fastapi.dependencies.utils'
```

it means the installed FastAPI version is too new for LiteLLM. Fix it by downgrading FastAPI:

```bash
pip uninstall fastapi
pip install 'fastapi<0.121.0'
```

After this, LiteLLM should start normally.

---

### Cline setup

#### 1. Configure LiteLLM (`litellm-config.yaml`)

```yaml
model_list:
  - model_name: claude-opus-5
    litellm_params:
      model: openai/cline-pass/qwen3.8-max
      api_base: http://127.0.0.1:5001
      api_key: YOUR_CLINE_API_KEY

  - model_name: claude-sonnet-5
    litellm_params:
      model: openai/cline-pass/deepseek-v4-flash
      api_base: http://127.0.0.1:5001
      api_key: YOUR_CLINE_API_KEY

litellm_settings:
  master_key: sk-1234567890 # authentication for LiteLLM
  drop_params: true
  anthropic_route: true # enables the /v1/messages endpoint
  use_chat_completions_url_for_anthropic_messages: true # route via /chat/completions, not /responses
```

> **Important:** `api_base` points to the **local unwrap proxy** (`http://127.0.0.1:5001`), not directly to cline.bot. The proxy handles forwarding to cline.bot and unwrapping the response.

> **Note on model names:** The `model` values above (e.g. `openai/cline-pass/deepseek-v4-flash`) use the `cline-pass/` prefix, which is for users with **Cline Pass**. If you use Cline **without** Cline Pass, the model names won't have the `cline-pass/` prefix — they'll be something like `openai/deepseek/deepseek-v4-flash`. Adjust the `model` values in your config to match the models available on your account.

---

### OpenCode Go setup

#### 1. LiteLLM config (`litellm-config-opencode.yaml`)

Models that go through LiteLLM (OpenAI-compatible ones) are already configured:

```yaml
model_list:
  - model_name: claude-opus-5
    litellm_params:
      model: openai/deepseek-v4-pro
      api_base: https://opencode.ai/zen/go/v1
      api_key: YOUR_OPENCODE_API_KEY

  - model_name: claude-sonnet-5
    litellm_params:
      model: openai/deepseek-v4-flash
      api_base: https://opencode.ai/zen/go/v1
      api_key: YOUR_OPENCODE_API_KEY

litellm_settings:
  master_key: sk-1234567890
  drop_params: true
  anthropic_route: true
  use_chat_completions_url_for_anthropic_messages: true
```

The `model` field uses the bare model ID (e.g. `deepseek-v4-pro`) without any prefix.

#### 2. Routing proxy environment

The routing proxy (`opencode_proxy.py`) needs the **OpenCode API key** as an
environment variable:

```bash
set OPENCODE_API_KEY=your_opencode_api_key_here
```

It reads it from the `OPENCODE_API_KEY` env var. The proxy listens on port 4000
by default and LiteLLM runs on port 4001.

#### 3. Claude Code (`settings.json`)

Point Claude Code at the routing proxy:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4000",
    "ANTHROPIC_AUTH_TOKEN": "sk-1234567890"
  },
  "theme": "dark",
  "model": "sonnet"
}
```

> The routing proxy receives all requests. For Anthropic-native models (Qwen,
> MiniMax) it forwards directly to OpenCode. For others it sends to LiteLLM
> on port 4001 for translation.

#### 4. How it decides routing

The proxy checks the request body's `model` field against a built-in map in
`opencode_proxy.py`. If the mapped model ID is in `ANTHROPIC_NATIVE_MODELS`
(Qwen, MiniMax), it goes directly to OpenCode's `/v1/messages` — no LiteLLM
involved. Otherwise it goes through LiteLLM for format translation.

To add a custom mapping, edit the `MODEL_MAP` dict in `opencode_proxy.py`.

---

#### 2. Configure Claude Code (`settings.json`)

For Cline, place this in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4000",
    "ANTHROPIC_AUTH_TOKEN": "sk-1234567890"
  },
  "theme": "dark",
  "model": "sonnet"
}
```

- `ANTHROPIC_BASE_URL` → points to LiteLLM on port 4000
- `ANTHROPIC_AUTH_TOKEN` → must match LiteLLM's `master_key`
- `model` → the model name (must match a `model_name` in the LiteLLM config)

---

## How to Run

### Cline

#### 1. Install LiteLLM

```bash
pip install 'litellm[proxy]'
```

If you hit the FastAPI dependency issue described above, downgrade FastAPI after installing:

```bash
pip uninstall fastapi
pip install 'fastapi<0.121.0'
```

#### 2. Start the unwrap proxy

```bash
python3 cline_unwrap_proxy.py
```

The proxy listens on `127.0.0.1:5001` by default. To change the port:

```bash
PROXY_PORT=5002 python3 cline_unwrap_proxy.py
```

#### 3. Start LiteLLM

```bash
litellm --config litellm-config.yaml --port 4000
```

#### 4. Start Claude Code

```bash
claude
```

### OpenCode Go

#### 1. Set the API key

```bash
set OPENCODE_API_KEY=your_opencode_api_key_here
```

#### 2. Start LiteLLM (for OpenAI-compatible models)

```bash
litellm --config litellm-config-opencode.yaml --port 4001
```

#### 3. Start the routing proxy

```bash
python3 opencode_proxy.py
```

The proxy listens on port 4000 by default. It auto-detects which models go
directly to OpenCode (Qwen, MiniMax) and which route through LiteLLM.

#### 4. Start Claude Code

```bash
claude
```

Claude Code points at the routing proxy (port 4000), which handles everything.

---

## Verification

Test the full chain with a curl request to LiteLLM's `/v1/messages` endpoint:

```bash
curl -s http://localhost:4000/v1/messages \
  -H "Authorization: Bearer sk-1234567890" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-5",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

A successful response looks like:

```json
{
  "id": "gen_...",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-5",
  "usage": { "input_tokens": 6, "output_tokens": 50 },
  "content": [
    { "type": "thinking", "thinking": "...", "signature": null },
    { "type": "text", "text": "Hello!" }
  ],
  "stop_reason": "end_turn"
}
```

If you get a valid response with `"type": "message"`, everything is working.

---

## Troubleshooting

| Error                                                               | Cause                                                         | Fix                                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `RouterRateLimitError: No deployments available for selected model` | Deployment went into cooldown after a failed upstream request | Check the LiteLLM logs for the underlying error; run with `--detailed_debug`      |
| `404 Not Found` for `.../responses`                                 | LiteLLM used the Responses API instead of `/chat/completions` | Add `use_chat_completions_url_for_anthropic_messages: true` to `litellm_settings` |
| `provider returned a response with no 'choices'`                    | cline.bot wraps the response in `data`                        | Make sure LiteLLM's `api_base` points to the unwrap proxy (port 5001)             |
| `401 Unauthorized`                                                  | Wrong Cline API key                                           | Verify `api_key` in `litellm-config.yaml` (the proxy forwards it as-is)           |
| `This model isn't mapped yet`                                       | Model not in LiteLLM's pricing database                       | Harmless — only affects cost tracking, not functionality                          |
| `ImportError: cannot import name 'get_flat_dependant'`              | FastAPI version too new for LiteLLM                           | `pip uninstall fastapi && pip install 'fastapi<0.121.0'`                          |
| OpenCode direct route hangs or errors                               | Model not in `ANTHROPIC_NATIVE_MODELS` set in proxy           | Add the model id to `ANTHROPIC_NATIVE_MODELS` in `opencode_proxy.py`              |
| OpenCode chat route has wrong model ID                              | `MODEL_MAP` maps to wrong OpenCode model ID                  | Fix the mapping in `MODEL_MAP` in `opencode_proxy.py`                             |

### Debugging

Run LiteLLM with verbose logging to see the exact request/response:

```bash
litellm --config litellm-config.yaml --port 4000 --detailed_debug
```

---

## Files

| File                          | Purpose                                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| `litellm-config.yaml`         | LiteLLM proxy configuration for Cline backend                    |
| `litellm-config-opencode.yaml`| LiteLLM proxy configuration for OpenCode Go backend              |
| `settings.json`               | Claude Code configuration (env vars, model)                      |
| `cline_unwrap_proxy.py`       | Pass-through proxy that unwraps cline.bot's `data` field         |
| `opencode_proxy.py`           | Routing proxy that sends Anthropic-native models direct, others via LiteLLM |
| `dashboard.py`                | Web dashboard that manages the setup with both backends          |
| `dashboard_state.json`        | Dashboard state (persisted ports, backend selection)             |

---

## Web Dashboard

`dashboard.py` is a single-file, stdlib-only web dashboard that manages the whole
setup from a browser. It needs **no extra installs** (only `litellm` + Python).

### Run it

```bash
python3 dashboard.py
```

Then open **http://127.0.0.1:8080**. (Change the port with `DASH_PORT=9000 python3 dashboard.py`.)

### What it does

- **Backend selector** — switch between Cline and OpenCode Go backends. Each shows the relevant model lists (grouped by endpoint type for OpenCode), API key fields, and port configuration.
- **Creates `litellm-config.yaml`** on first launch if it doesn't exist.
- **Edit models & keys** — pick the Opus and Sonnet models from dropdowns, or choose **"Custom…"** to type your own model id. For OpenCode, models are grouped by endpoint type (`/v1/chat/completions`, `/v1/messages`).
- **Start / Stop proxy** — one button launches the whole chain for the selected backend:
  - Cline: `litellm → cline_unwrap_proxy.py → cline.bot`
  - OpenCode: `routing proxy + litellm → opencode.ai`
- **Auto-restart on change** — whenever you save the config while the proxy is
  running, the chain is stopped and restarted with the new settings automatically.
- **Live logs** — a tabbed, auto-refreshing log viewer for all running processes.

### How it stores settings

- The **models, API keys, and litellm settings** are written to `litellm-config.yaml`
  (for Cline) or `litellm-config-opencode.yaml` (for OpenCode Go).
- The **backend selection, litellm port, and proxy port** are kept in `dashboard_state.json`.

### Manual steps it automates

These are the same steps as the [How to Run](#how-to-run) section — the dashboard's
**Start proxy** button runs the full chain for the selected backend:

**Cline:**
1. Starts `cline_unwrap_proxy.py` (port from your config, default 5001).
2. Starts `litellm --config litellm-config.yaml --port <litellm_port>`.

**OpenCode Go:**
1. Starts LiteLLM with `--config litellm-config-opencode.yaml --port 4001`.
2. Starts `opencode_proxy.py` (port 4000), which routes requests between LiteLLM and OpenCode directly.

Then point Claude Code at the proxy (port 4000 for either backend via the `settings.json` hint in the UI).
