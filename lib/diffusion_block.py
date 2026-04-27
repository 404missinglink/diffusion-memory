import numpy as np
from scipy.sparse import csr_matrix


class DiffusionBlock:
    """Core diffusion update: s <- s + eta(Ps - s) - leak*s."""

    def __init__(self, transition_matrix: csr_matrix):
        self.transition_matrix = transition_matrix

    def step(self, strengths: np.ndarray, diffusion_rate: float, leak_rate: float) -> np.ndarray:
        """Apply one diffusion+leak update and clamp to valid strength range."""
        # Diffuse over graph, then apply uniform leakage.
        ps = self.transition_matrix @ strengths
        strengths = strengths + diffusion_rate * (ps - strengths) - leak_rate * strengths
        return np.clip(strengths, 0.0, 1.0)
