# Claude Code → Cline API via LiteLLM

This project connects **Claude Code** (Anthropic's CLI coding agent) to the **Cline API** (api.cline.bot) using **LiteLLM** as a proxy. It lets you use Cline's models (e.g. DeepSeek, Qwen) from within Claude Code by translating the Anthropic `messages` API to OpenAI's `chat/completions` API.

## Request flow

1. Claude Code sends an **Anthropic**-format request to `POST /v1/messages` on LiteLLM.
2. LiteLLM translates it to the **OpenAI** format and forwards it as `POST /chat/completions`.
3. The **unwrap proxy** (`cline_unwrap_proxy.py`) forwards it to cline.bot and unwraps the response.
4. cline.bot returns the completion, which flows back up to Claude Code.

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

## Prerequisites

- **Python 3.7+**
- **LiteLLM** installed (see [How to Run](#how-to-run))
- A **Cline API key** from [cline.bot](https://cline.bot)
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

## Setup

### 1. Configure LiteLLM (`litellm-config.yaml`)

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

### 2. Configure Claude Code (`settings.json`)

Place this in `~/.claude/settings.json`:

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

### 1. Install LiteLLM

```bash
pip install 'litellm[proxy]'
```

If you hit the FastAPI dependency issue described above, downgrade FastAPI after installing:

```bash
pip uninstall fastapi
pip install 'fastapi<0.121.0'
```

### 2. Start the unwrap proxy

The proxy reads the API key from the `Authorization` header that LiteLLM forwards (which comes from `api_key` in `litellm-config.yaml`), so no `CLINE_API_KEY` env var is needed:

```bash
python3 cline_unwrap_proxy.py
```

The proxy listens on `127.0.0.1:5001` by default. To change the port:

```bash
PROXY_PORT=5002 python3 cline_unwrap_proxy.py
```

### 3. Start LiteLLM

```bash
litellm --config litellm-config.yaml --port 4000
```

LiteLLM listens on `http://localhost:4000`.

### 4. Start Claude Code

```bash
claude
```

Claude Code will now use the models you configured in LiteLLM.

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

### Debugging

Run LiteLLM with verbose logging to see the exact request/response:

```bash
litellm --config litellm-config.yaml --port 4000 --detailed_debug
```

---

## Files

| File                    | Purpose                                                  |
| ----------------------- | -------------------------------------------------------- |
| `litellm-config.yaml`   | LiteLLM proxy configuration (models, settings)           |
| `settings.json`         | Claude Code configuration (env vars, model)              |
| `cline_unwrap_proxy.py` | Pass-through proxy that unwraps cline.bot's `data` field |
| `dashboard.py`          | Web dashboard that manages the setup below               |

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

- **Creates `litellm-config.yaml`** on first launch if it doesn't exist (with
  defaults you can edit in the UI).
- **Edit models & keys** — pick the Opus and Sonnet models from a dropdown of the
  ClinePass models (at [cline.bot docs](https://docs.cline.bot/getting-started/clinepass#models)),
  or choose **"Custom…"** to type your own model id. Edit the Cline API key and
  the LiteLLM master key in plain-text fields.
- **Start / Stop proxy** — one button launches the whole chain:
  `litellm → cline_unwrap_proxy.py → cline.bot`.
- **Auto-restart on change** — whenever you save the config while the proxy is
  running, the chain is stopped and restarted with the new settings automatically.
- **Live logs** — a tabbed, auto-refreshing log viewer for both the litellm proxy
  and the unwrap proxy.
- **Settings hint** — shows the matching `~/.claude/settings.json` snippet for
  your current master key and litellm port.

### How it stores settings

- The **models, API keys, and litellm settings** are written to `litellm-config.yaml`
  (the same file the manual workflow uses).
- The **litellm port** is kept in `dashboard_state.json` next to the config, because
  `litellm-config.yaml` has no port entry — the dashboard persists it so the port you
  set in the UI is the port the proxy runs on.

### Manual steps it automates

These are the same steps as the [How to Run](#how-to-run) section — the dashboard's
**Start proxy** button just runs them for you:

1. Starts `cline_unwrap_proxy.py` (port from your config, default 5001).
2. Starts `litellm --config litellm-config.yaml --port <litellm_port>`.

Then point Claude Code at LiteLLM (step 4 above / the `settings.json` hint in the UI).
