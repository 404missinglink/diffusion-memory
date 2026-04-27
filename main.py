"""
Gradient Streaming Memory (Real Graph Diffusion)
------------------------------------------------
Canonical implementation in this file is a discrete graph diffusion block
over token memories:
    s_{t+1} = s_t + eta * (P s_t - s_t) - leak * s_t + input_t
where P is a row-stochastic token graph transition matrix.

Asynchronous updates:
- each timestep injects new stimulus tokens at age t=0
- existing memories evolve only via the diffusion block above
- after stimulus stops, we track when total amnesia occurs

Install deps:
    uv pip install datasets scikit-learn matplotlib
"""

import argparse
import warnings

from lib.diffusion_memory import DiffusionMemory
from lib.io_utils import load_texts
from lib.plotting import plot_results
from lib.run_utils import (
    apply_sim_overrides,
    build_metrics,
    print_run_stats,
    save_metrics,
    validate_sim_params,
)

warnings.filterwarnings("ignore")


def parse_args():
    # Keep the CLI explicit so experiments are reproducible.
    parser = argparse.ArgumentParser(description="Run graph-diffusion token memory simulation.")
    parser.add_argument("--dataset", default="ag_news", help="HuggingFace dataset name")
    parser.add_argument("--split", default="train[:300]", help="HuggingFace split expression")
    parser.add_argument("--chat-json", default=None, help="Path to JSON file with list of {role, content} turns")
    parser.add_argument("--continuous-stimulus", action="store_true", help="Inject new stimuli at every timestep")
    parser.add_argument("--stimulus-steps", type=int, default=None, help="Override stimulus steps")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total simulation steps")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--diffusion-rate", type=float, default=None, help="Override diffusion rate")
    parser.add_argument("--leak-rate", type=float, default=None, help="Override leak rate")
    parser.add_argument("--amnesia-threshold", type=float, default=None, help="Override alive threshold")
    parser.add_argument("--stimulus-seed", type=int, default=123, help="Seed for stimulus order")
    parser.add_argument("--plot-out", default="outputs/gradient_memory_plot.png", help="Output plot path")
    parser.add_argument("--metrics-out", default="outputs/last_run_metrics.json", help="Output JSON metrics path")
    return parser.parse_args()


def main():
    # 1) Load corpus/chat inputs.
    args = parse_args()
    texts = load_texts(args)

    # 2) Build memory state and score token importance.
    print("Building token-level memory...")
    memory = DiffusionMemory(texts)

    print("Scoring token importance...")
    scored = memory.score_importance()
    sim_params = memory.default_simulation_params()
    apply_sim_overrides(sim_params, args)
    validate_sim_params(sim_params)

    # 3) Run the diffusion simulation.
    diff = memory.simulate_graph_diffusion_memory(
        scored["importance"],
        continuous_stimulus=args.continuous_stimulus,
        stimulus_seed=args.stimulus_seed,
        **sim_params,
    )

    # 4) Report key run statistics.
    print("\nAuto-scaled simulation params:")
    print(
        f"  stimulus_steps={sim_params['stimulus_steps']}, "
        f"max_total_steps={sim_params['max_total_steps']}, "
        f"continuous_stimulus={args.continuous_stimulus}, "
        f"diffusion_rate={sim_params['diffusion_rate']:.4f}, "
        f"leak_rate={sim_params['leak_rate']:.4f}"
    )

    print_run_stats(memory, diff)

    # 5) Render diagnostics if simulation produced any steps.
    if diff["t_steps"] > 0:
        print("\nGenerating plots...")
        plot_results(memory.n_tokens, scored["signals_norm"], diff, args.plot_out)
        print(f"\nPlot saved -> {args.plot_out}")
    else:
        print("\nSkipping plots: no simulation steps were executed.")

    # 6) Persist machine-readable metrics for later comparison.
    metrics = build_metrics(args, texts, memory, sim_params, diff)
    save_metrics(args.metrics_out, metrics)
    print(f"Metrics saved -> {args.metrics_out}")


if __name__ == "__main__":
    main()