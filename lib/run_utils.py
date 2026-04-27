import json
import os

import numpy as np


def apply_sim_overrides(sim_params: dict, args):
    """Apply CLI-provided overrides on top of model defaults."""
    # Override only values explicitly provided by the caller.
    overrides = {
        "stimulus_steps": args.stimulus_steps,
        "max_total_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "diffusion_rate": args.diffusion_rate,
        "leak_rate": args.leak_rate,
        "amnesia_threshold": args.amnesia_threshold,
    }
    for key, value in overrides.items():
        if value is not None:
            sim_params[key] = value


def validate_sim_params(sim_params: dict):
    """Validate numeric simulation parameters before running."""
    # Basic guardrails to avoid invalid dynamics.
    checks = (
        ("max_total_steps", lambda v: v > 0, "must be > 0"),
        ("stimulus_steps", lambda v: v >= 0, "must be >= 0"),
        ("learning_rate", lambda v: v >= 0, "must be >= 0"),
        ("diffusion_rate", lambda v: v >= 0, "must be >= 0"),
        ("leak_rate", lambda v: v >= 0, "must be >= 0"),
        ("amnesia_threshold", lambda v: v >= 0, "must be >= 0"),
    )
    for key, ok, msg in checks:
        if not ok(sim_params[key]):
            raise ValueError(f"{key} {msg}")


def top_survivor_indices(scores: np.ndarray, survived_mask: np.ndarray, k: int) -> np.ndarray:
    # Return top-k survivors sorted by score (high -> low).
    survived_idx = np.where(survived_mask)[0]
    return np.array([], dtype=int) if survived_idx.size == 0 else survived_idx[np.argsort(scores[survived_idx])[-k:]][::-1]


def print_run_stats(memory, diff: dict):
    # Human-readable run summary for CLI usage.
    survived_mask = diff["survived_mask"]
    scores = diff["scores"]
    print(f"\nTokens alive at final step: {survived_mask.sum()} / {memory.n_tokens}")
    print(f"Stimulus steps: {diff['stimulus_steps']}")
    if diff["amnesia_step"] is None:
        print(f"Total amnesia: not reached by step {diff['max_total_steps']}")
    else:
        print(f"Total amnesia reached at timestep: {diff['amnesia_step']}")
        print(f"N steps until amnesia after stimulus: {diff['amnesia_step'] - diff['stimulus_steps']}")
    print("\nTop surviving tokens:")
    top_idx = top_survivor_indices(scores, survived_mask, k=5)
    if len(top_idx) == 0:
        print("  No survivors at final timestep.")
    else:
        for i in top_idx:
            tok_id = memory.idx_to_token[i]
            tok_text = memory._token_label(tok_id).replace("\n", "\\n")
            print(f"  [{scores[i]:.3f}] id={tok_id} token={tok_text!r}")


def build_metrics(args, texts: list, memory, sim_params: dict, diff: dict) -> dict:
    # Structured metrics payload for reproducible experiment tracking.
    survived_mask = diff["survived_mask"]
    scores = diff["scores"]
    return {
        "dataset": args.dataset,
        "split": args.split,
        "chat_json": args.chat_json,
        "n_docs": len(texts),
        "n_tokens": memory.n_tokens,
        "sim_params": sim_params,
        "continuous_stimulus": args.continuous_stimulus,
        "stimulus_seed": args.stimulus_seed,
        "final_alive": int(survived_mask.sum()),
        "amnesia_step": diff["amnesia_step"],
        "steps_after_stim_to_amnesia": None
        if diff["amnesia_step"] is None
        else int(diff["amnesia_step"] - diff["stimulus_steps"]),
        "top_survivors": [
            {
                "score": float(scores[i]),
                "token_id": int(memory.idx_to_token[i]),
                "token": memory._token_label(memory.idx_to_token[i]).replace("\n", "\\n"),
            }
            for i in top_survivor_indices(scores, survived_mask, k=10)
        ],
    }


def save_metrics(path: str, metrics: dict):
    """Write metrics JSON to disk."""
    # Ensure destination exists before writing JSON.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
