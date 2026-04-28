# LongMemEval Pacabench Run - 2026-04-28

## Summary

- Run ID: `diffusion-memory-longmemeval-5QLL1A0D`
- Benchmark: `diffusion-memory-longmemeval`
- Dataset: `longmemeval-oracle`
- Dataset source: `huggingface:xiaowu0162/longmemeval-cleaned`
- Split: `longmemeval_oracle`
- Agent: `diffusion-memory`
- Agent model: `gpt-5-nano-2025-08-07`
- Judge model: `gpt-4o-mini`
- Cases: `500/500`
- Passed: `349`
- Failed: `151`
- Accuracy: `69.8%`
- System errors: `0`
- Total wall time: `76m30s`
- Reported cost: `$0.0132`

## Command

The run was executed from the repository root with an OpenAI API key supplied
only through the process environment:

```bash
export OPENAI_API_KEY=...
uv run pacabench run --no-tui
```

The local environment was prepared with:

```bash
uv sync --dev
```

## Performance

- Duration p50: `6.6s`
- Duration p95: `19.7s`
- Max case duration: `49.6s`
- LLM latency average: `8.4s`
- LLM latency p50: `6.5s`
- LLM latency p95: `19.7s`
- Attempts average: `1.0`
- Attempts max: `1`

## Token Usage

- Agent input tokens: `2,086,280`
- Agent output tokens: `453,879`
- Agent LLM calls: `500`
- Judge input tokens: `85,899`
- Judge output tokens: `500`

## Cost

- Total: `$0.0132`
- Agent: `$0.0000`
- Judge: `$0.0132`

## Artifacts

The run artifacts are committed under:

```text
runs/diffusion-memory-longmemeval-5QLL1A0D/
```

Important files:

- `runs/diffusion-memory-longmemeval-5QLL1A0D/metadata.json`
- `runs/diffusion-memory-longmemeval-5QLL1A0D/pacabench.yaml`
- `runs/diffusion-memory-longmemeval-5QLL1A0D/results.jsonl`

No `system_errors.jsonl` file was produced for this run.

## Validation

After the run completed, the local test suite was executed:

```bash
uv run pytest
```

Result:

```text
3 passed
```
