import os

import matplotlib.pyplot as plt
import numpy as np


def plot_results(n_items: int, signals_norm: np.ndarray, diff: dict, out_path: str):
    # Multi-panel diagnostic figure for distribution, retention, rank, and signal separation.
    n = n_items
    score_history = diff["score_history"]
    survived_mask = diff["survived_mask"]
    scores = diff["scores"]
    t_steps = diff["t_steps"]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d0d0d")

    ax1 = fig.add_subplot(2, 3, (1, 2))
    ax1.set_facecolor("#0d0d0d")
    cmap = plt.cm.plasma
    # Plot a small set of snapshots to keep the histogram panel readable.
    n_snapshots = min(8, t_steps)
    snapshot_steps = np.unique(np.linspace(0, t_steps - 1, n_snapshots, dtype=int))
    for t in snapshot_steps:
        color = cmap(t / max(1, t_steps - 1))
        ax1.hist(score_history[t], bins=30, alpha=0.4, color=color, label=f"t={t+1}", density=True)
    ax1.set_title("Importance Score Distribution Over Timesteps", color="white", fontsize=12)
    ax1.set_xlabel("Importance Score", color="#aaaaaa")
    ax1.set_ylabel("Density", color="#aaaaaa")
    ax1.tick_params(colors="#aaaaaa")
    ax1.spines[:].set_color("#333333")
    ax1.legend(fontsize=7, labelcolor="white", facecolor="#1a1a1a")

    ax2 = fig.add_subplot(2, 3, 3)
    ax2.set_facecolor("#0d0d0d")
    # Retention curve shows how many tokens remain above threshold over time.
    retention = [m.sum() / n * 100 for m in diff["alive_history"]]
    ax2.plot(range(1, t_steps + 1), retention, color="#ff6b6b", linewidth=2.5, label="Retention")
    ax2.fill_between(range(1, t_steps + 1), retention, alpha=0.15, color="#ff6b6b")
    stim_end = diff["stimulus_steps"]
    ax2.axvline(stim_end, color="#4ecdc4", linestyle="--", linewidth=1.5, label="Stimulus stops")
    if diff["amnesia_step"] is not None:
        ax2.axvline(diff["amnesia_step"], color="#ffd166", linestyle=":", linewidth=1.8, label="Total amnesia")
    ax2.set_title("Streaming Memory Retention", color="white", fontsize=12)
    ax2.set_xlabel("Diffusion Timestep", color="#aaaaaa")
    ax2.set_ylabel("% Tokens Alive", color="#aaaaaa")
    ax2.tick_params(colors="#aaaaaa")
    ax2.spines[:].set_color("#333333")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a1a1a")

    ax3 = fig.add_subplot(2, 3, (4, 5))
    ax3.set_facecolor("#0d0d0d")
    # Rank plot emphasizes score concentration and survivor tail.
    rank_order = np.argsort(scores)[::-1]
    ranked_scores = scores[rank_order]
    survivor_rank_mask = survived_mask[rank_order]
    ax3.plot(np.arange(1, n + 1), ranked_scores, color="#aaaaaa", linewidth=1.0, alpha=0.7)
    ax3.scatter(
        np.where(survivor_rank_mask)[0] + 1,
        ranked_scores[survivor_rank_mask],
        color="#4ecdc4",
        s=20,
        alpha=0.85,
        label="Survived",
    )
    ax3.set_title("Final Token Strength by Rank", color="white", fontsize=12)
    ax3.set_xlabel("Rank (high to low score)", color="#aaaaaa")
    ax3.set_ylabel("Final Score", color="#aaaaaa")
    ax3.tick_params(colors="#aaaaaa")
    ax3.spines[:].set_color("#333333")
    ax3.legend(fontsize=8, labelcolor="white", facecolor="#1a1a1a")

    ax4 = fig.add_subplot(2, 3, 6)
    ax4.set_facecolor("#0d0d0d")
    # Compare average input signals for kept vs pruned tokens.
    signal_names = ["Frequency", "IDF"] if signals_norm.shape[1] == 2 else ["Frequency", "IDF", "Signal 3"]
    if survived_mask.any() and (~survived_mask).any():
        survived_signals = signals_norm[survived_mask].mean(axis=0)
        pruned_signals = signals_norm[~survived_mask].mean(axis=0)
    else:
        survived_signals = signals_norm.mean(axis=0)
        pruned_signals = np.zeros_like(survived_signals)
    x = np.arange(len(signal_names))
    w = 0.35
    ax4.bar(x - w / 2, survived_signals, w, label="Survived", color="#4ecdc4", alpha=0.85)
    ax4.bar(x + w / 2, pruned_signals, w, label="Pruned", color="#ff6b6b", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(signal_names, color="#aaaaaa", fontsize=9)
    ax4.set_title("Avg Signal: Survived vs Pruned", color="white", fontsize=12)
    ax4.set_ylabel("Normalized Score", color="#aaaaaa")
    ax4.tick_params(colors="#aaaaaa")
    ax4.spines[:].set_color("#333333")
    ax4.legend(fontsize=8, labelcolor="white", facecolor="#1a1a1a")

    plt.suptitle("Asynchronous Streaming Token Memory", color="white", fontsize=15, y=1.01)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
