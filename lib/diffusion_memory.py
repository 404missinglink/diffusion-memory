import tiktoken
import numpy as np
from scipy.sparse import csr_matrix

from lib.diffusion_graph import DiffusionGraphBuilder
from lib.diffusion_simulator import DiffusionSimulator
from lib.importance_model import ImportanceModel


class DiffusionMemory:
    """High-level orchestrator for scoring tokens and running diffusion dynamics."""

    # Sparse neighbor budget per token in the transition graph.
    GRAPH_TOP_K = 12

    def __init__(self, texts: list):
        # Tokenize and prepare corpus-level sparse matrices once up front.
        self.texts = texts
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.docs = self._build_docs(texts)
        self.doc_count = len(self.docs)
        self.vocab = sorted({tok for doc in self.docs for tok in doc})
        self.n_tokens = len(self.vocab)
        self.token_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.idx_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.indexed_docs = [
            np.fromiter((self.token_to_idx[tok] for tok in doc), dtype=int, count=len(doc))
            for doc in self.docs
        ]
        self.doc_token_counts, self.doc_presence = self._build_doc_matrices()
        # Derived statistics reused by the scoring model.
        self.term_freq = np.asarray(self.doc_token_counts.sum(axis=0)).ravel()
        self.doc_freq = np.asarray(self.doc_presence.sum(axis=0)).ravel()
        self.transition_matrix = DiffusionGraphBuilder(
            self.doc_presence,
            self.GRAPH_TOP_K,
        ).build_transition_matrix(self.n_tokens)
        self.simulator = DiffusionSimulator(
            self.indexed_docs,
            self.transition_matrix,
            self.n_tokens,
            self.doc_count,
        )

    def _build_docs(self, items: list):
        """Extract raw text from inputs and tokenize in batch."""
        # Batch tokenization is faster than per-item encode calls.
        texts = [str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in items]
        return self.encoding.encode_batch(texts)

    def _build_doc_matrices(self) -> tuple[csr_matrix, csr_matrix]:
        # Build document-token count and binary presence matrices in CSR format.
        if self.doc_count == 0 or self.n_tokens == 0:
            shape = (self.doc_count, self.n_tokens)
            empty = csr_matrix(shape, dtype=float)
            return empty, empty.copy()

        lengths = np.array([doc.size for doc in self.indexed_docs], dtype=int)
        indptr = np.concatenate(([0], np.cumsum(lengths)))
        indices = np.concatenate(self.indexed_docs) if lengths.sum() > 0 else np.array([], dtype=int)
        data = np.ones(indices.size, dtype=float)
        counts = csr_matrix((data, indices, indptr), shape=(self.doc_count, self.n_tokens), dtype=float)
        counts.sum_duplicates()
        # Presence matrix (0/1) is used for document-frequency and co-occurrence.
        presence = counts.sign().tocsr()
        return counts, presence

    def default_simulation_params(self) -> dict:
        """Return simulation defaults scaled by corpus size."""
        # Simple size-scaled defaults with fixed dynamics constants.
        stimulus_steps = max(30, min(self.doc_count, int(0.4 * self.doc_count)))
        return {
            "stimulus_steps": int(stimulus_steps),
            "max_total_steps": max(stimulus_steps + 300, int(stimulus_steps * 8)),
            "learning_rate": 0.75,
            "diffusion_rate": 0.20,
            "leak_rate": 0.01,
            "amnesia_threshold": 0.02,
        }

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize one text string using the configured encoding."""
        return self.encoding.encode(text)

    def _token_label(self, token_id: int) -> str:
        # Decode individual token bytes safely for debugging/metrics output.
        raw = self.encoding.decode_single_token_bytes(token_id)
        return raw.decode("utf-8", errors="replace")

    def score_importance(self):
        """Compute normalized token importance from TF and document frequency."""
        return ImportanceModel.score_from_arrays(self.term_freq, self.doc_freq, self.doc_count)

    def simulate_graph_diffusion_memory(
        self,
        importance,
        stimulus_steps: int = 120,
        max_total_steps: int = 5000,
        continuous_stimulus: bool = False,
        stimulus_seed: int = 123,
        learning_rate: float = 0.75,
        diffusion_rate: float = 0.20,
        leak_rate: float = 0.01,
        amnesia_threshold: float = 0.02,
    ):
        """Run the diffusion simulation and return timestep histories + final state."""
        return self.simulator.run(
            importance=importance,
            stimulus_steps=stimulus_steps,
            max_total_steps=max_total_steps,
            continuous_stimulus=continuous_stimulus,
            stimulus_seed=stimulus_seed,
            learning_rate=learning_rate,
            diffusion_rate=diffusion_rate,
            leak_rate=leak_rate,
            amnesia_threshold=amnesia_threshold,
        )
