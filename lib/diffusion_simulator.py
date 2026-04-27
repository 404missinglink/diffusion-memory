import numpy as np
from scipy.sparse import csr_matrix

from lib.diffusion_block import DiffusionBlock


class DiffusionSimulator:
    """Time-stepped simulator that applies stimulus, diffusion, and leakage."""
    def __init__(
        self,
        indexed_docs: list[np.ndarray],
        transition_matrix: csr_matrix,
        n_tokens: int,
        doc_count: int,
    ):
        self.indexed_docs = indexed_docs
        self.n_tokens = n_tokens
        self.doc_count = doc_count
        self.diffusion_block = DiffusionBlock(transition_matrix)

    def _inject_stimulus(
        self,
        strengths: np.ndarray,
        doc_idx: int,
        importance: np.ndarray,
        learning_rate: float,
    ) -> np.ndarray:
        """Boost strengths for tokens present in the current stimulus document."""
        # Inject current document tokens into strengths at t=0 age.
        if self.indexed_docs[doc_idx].size == 0:
            return np.zeros(self.n_tokens, dtype=bool)

        unique_idx = np.unique(self.indexed_docs[doc_idx])
        stimulus_mask = np.zeros(self.n_tokens, dtype=bool)
        stimulus_mask[unique_idx] = True
        strengths[unique_idx] = np.clip(
            strengths[unique_idx] + learning_rate * importance[unique_idx],
            0.0,
            1.0,
        )
        return stimulus_mask

    def run(
        self,
        importance: np.ndarray,
        stimulus_steps: int = 120,
        max_total_steps: int = 5000,
        continuous_stimulus: bool = False,
        stimulus_seed: int = 123,
        learning_rate: float = 0.75,
        diffusion_rate: float = 0.20,
        leak_rate: float = 0.01,
        amnesia_threshold: float = 0.02,
    ):
        """Execute simulation loop and capture full histories for analysis/plots."""
        # Handle empty vocab gracefully with shape-compatible outputs.
        if self.n_tokens == 0:
            return {
                "alive_history": [],
                "score_history": np.zeros((0, 0), dtype=float),
                "stimulus_history": [],
                "scores": np.array([], dtype=float),
                "survived_mask": np.array([], dtype=bool),
                "t_steps": 0,
                "stimulus_steps": stimulus_steps,
                "continuous_stimulus": continuous_stimulus,
                "amnesia_step": None,
                "max_total_steps": max_total_steps,
            }

        strengths = np.zeros(self.n_tokens, dtype=float)
        alive_history, score_history, stimulus_history = [], [], []

        # Shuffle document order once for deterministic but nontrivial stimulation.
        stim_rng = np.random.default_rng(stimulus_seed)
        doc_order = np.arange(self.doc_count)
        stim_rng.shuffle(doc_order)

        amnesia_step = None
        for t in range(max_total_steps):
            # External forcing phase.
            if continuous_stimulus or t < stimulus_steps:
                doc_idx = int(doc_order[t % len(doc_order)])
                stimulus_mask = self._inject_stimulus(strengths, doc_idx, importance, learning_rate)
            else:
                stimulus_mask = np.zeros(self.n_tokens, dtype=bool)

            # Internal memory dynamics phase.
            strengths = self.diffusion_block.step(strengths, diffusion_rate, leak_rate)
            alive = strengths > amnesia_threshold
            alive_history.append(alive.copy())
            score_history.append(strengths.copy())
            stimulus_history.append(stimulus_mask.copy())

            # First time all tokens fall below threshold after stimulus is amnesia.
            if t >= stimulus_steps and amnesia_step is None and alive.sum() == 0:
                amnesia_step = t + 1
                break

        score_history = np.array(score_history)
        return {
            "alive_history": alive_history,
            "score_history": score_history,
            "stimulus_history": stimulus_history,
            "scores": score_history[-1],
            "survived_mask": alive_history[-1],
            "t_steps": len(alive_history),
            "stimulus_steps": stimulus_steps,
            "continuous_stimulus": continuous_stimulus,
            "amnesia_step": amnesia_step,
            "max_total_steps": max_total_steps,
        }
