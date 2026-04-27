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

## TODO

- adding feature extractors (over just token level) 

- harness for LLM's?
