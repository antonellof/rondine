# Coding clients with Rondine

Rondine launches an **OpenAI-compatible** server from llama.cpp, MLX-LM, or vLLM.
Point any coding agent that speaks the Chat Completions API at the local base URL.

## Default endpoint

```text
http://127.0.0.1:8080/v1
```

API key: any non-empty string (most local servers ignore auth), e.g. `rondine`.

## Launch a coding preset

```bash
rondine plan auto --profile coding
rondine setup
rondine pull
rondine serve --profile coding
rondine verify --profile coding
```

`--profile coding` applies model-specific sampling (for example Qwen3.6 `temperature=0.6`)
and a practical **32K** context. Raise context only when you have spare unified memory / VRAM:

```bash
rondine plan qwen3.6-35b-a3b --profile coding --context 65536
```

## Cursor / Continue / OpenAI SDK

### Environment

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=rondine
```

### Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="rondine")
completion = client.chat.completions.create(
    model="rondine/qwen3.6-35b-a3b",  # or the id from GET /v1/models
    messages=[{"role": "user", "content": "Write a Rust fibonacci function."}],
)
print(completion.choices[0].message.content)
```

### curl

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rondine/qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Explain async vs threads in Python."}],
    "temperature": 0.6
  }'
```

## Model tips for code

| Machine | Suggested model |
|---|---|
| Mac 24–48GB | `qwen3.6-27b` or `gemma-4-12b` |
| Mac 48GB+ / Spark | `qwen3.6-35b-a3b` |
| Mac 128GB+ | `deepseek-v4-flash` (opt-in) |
| Mac 256GB / dual-node | `glm-5.2` (opt-in, low context) |

Rondine does **not** ship a proprietary coding agent loop. Use Cursor, Continue, Aider,
Codex CLI, Claude Code (custom base URL), or similar against the local server.

## Verify

```bash
rondine verify --profile coding
```

Checks `/health` or `/v1/models`, a short code-generation prompt, and a best-effort tool-call smoke.
