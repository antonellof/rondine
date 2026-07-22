# Benchmarks

Benchmarks are smoke-test baselines, not cross-platform rankings. Results vary
with model, quantization, context, prompt, backend version, thermals, and
concurrent load.

## Apple M2 Pro small-model baseline

Recorded on 2026-07-22:

- Hardware: Apple M2 Pro, 32GB unified memory, 10 CPU threads reported
- Backend: llama.cpp build 7650 (`68b4d516c`), Metal
- Model: `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF`
- Quant/file: Q4_K_M, 1.95GiB, GGUF v3
- Server context: 32,768 tokens
- Offload: 37/37 layers to Metal
- Projected device memory: 3,291MiB
- KV cache: 1,152MiB

Rondine verification passed:

- `GET /health`
- `GET /v1/models`
- Python code generation
- OpenAI-compatible function/tool calling

The decode benchmark sent the same request three times with temperature 0 and
`max_tokens=128`:

```text
Write a typed Python binary search function. Code only.
```

Observed decode rates were 64.4, 66.6, and 66.7 tokens/second, with a median of
66.6 tokens/second. Corresponding wall times were 2.02, 1.94, and 1.94 seconds.

## Reproduce

Start and verify the curated small model:

```bash
rondine plan qwen2.5-coder-3b --context 4096 --save-as small-coder
rondine pull qwen2.5-coder-3b
rondine serve --preset small-coder
rondine verify --name small-coder
```

Send a benchmark request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rondine/qwen2.5-coder-3b",
    "messages": [{
      "role": "user",
      "content": "Write a typed Python binary search function. Code only."
    }],
    "max_tokens": 128,
    "temperature": 0
  }'
```

llama.cpp includes prompt and generation timings in the response. Use
`timings.predicted_per_second` for decode throughput, repeat several times, and
report the median rather than a single run.

Stop the managed server afterward:

```bash
rondine stop --name small-coder
```

## Limitations

- One machine and one quantization were measured.
- The run measured single-request decode throughput, not concurrency or prefill.
- It is not a model-quality evaluation.
- The downloaded model and backend were warm by the timed requests.
