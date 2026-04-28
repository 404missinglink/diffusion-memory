# Diffusion Memory

A small token-memory simulation that spreads token strength over a graph (`diffusion`) while applying decay (`leak`) until memories fade (`amnesia`).

## Quick Start

```bash
uv pip install datasets scikit-learn matplotlib tiktoken scipy pytest
uv run python main.py --split "train[:80]" --max-steps 500
```

## Use With Chat JSON

```bash
uv run python main.py --chat-json path/to/chat.json --max-steps 500
```

Expected chat format:

```json
[
  { "role": "user", "content": "Hello" },
  { "role": "assistant", "content": "Hi there" }
]
```

## What You Get

When you run it, it will:

- Print simulation stats (alive tokens, amnesia step, top survivors)
- Save a plot to `outputs/gradient_memory_plot.png`
- Save metrics JSON to `outputs/last_run_metrics.json`

If amnesia is reached, you will see:

```text
Total amnesia reached at timestep: <N>
```

## Pacabench LongMemEval

This repo includes a Pacabench agent that uses diffusion-memory retrieval to
select relevant LongMemEval conversation turns, then answers with an
OpenAI-compatible chat model.

```bash
uv sync --dev
export OPENAI_API_KEY=...
uv run pacabench run --limit 1 --no-tui
```

For a fuller run:

```bash
uv run pacabench run --no-tui
```

The agent defaults to `gpt-5-nano`. You can override the model and retrieval
budget through environment variables:

```bash
OPENAI_MODEL=gpt-4o-mini \
DIFFUSION_CONTEXT_MESSAGES=32 \
DIFFUSION_CONTEXT_CHARS=32000 \
uv run pacabench run --limit 10 --no-tui
```

## TODO

- adding feature extractors (over just token level) 

- harness for LLM's?
