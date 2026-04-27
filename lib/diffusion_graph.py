import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class DiffusionGraphBuilder:
    """Builds a sparse row-stochastic transition matrix for graph diffusion."""
    def __init__(self, doc_presence: csr_matrix, top_k: int):
        self.doc_presence = doc_presence
        self.top_k = top_k

    @staticmethod
    def _add_self_loops_for_empty_rows(
        transition: csr_matrix,
        n_tokens: int,
        non_empty_rows: set[int],
    ) -> csr_matrix:
        """Ensure isolated rows have self-probability so every row can normalize."""
        empty_rows = np.array([i for i in range(n_tokens) if i not in non_empty_rows], dtype=int)
        if empty_rows.size == 0:
            return transition
        return transition + csr_matrix(
            (np.ones(empty_rows.size, dtype=float), (empty_rows, empty_rows)),
            shape=(n_tokens, n_tokens),
        )

    def build_transition_matrix(self, n_tokens: int) -> csr_matrix:
        """Construct token-token transition probabilities from document co-occurrence."""
        # Degenerate case: identity transition if no data exists.
        if self.doc_presence.shape[0] == 0 or n_tokens == 0:
            return csr_matrix(np.eye(n_tokens, dtype=float))

        # Inverse document frequency-like downweight on destination nodes.
        token_counts = np.asarray(self.doc_presence.sum(axis=0)).ravel()
        inv_token_counts = np.divide(
            1.0,
            np.maximum(token_counts, 1.0),
            out=np.zeros_like(token_counts, dtype=float),
            where=token_counts > 0,
        )

        # Token co-occurrence from doc-token incidence.
        cooc = (self.doc_presence.T @ self.doc_presence).tocsr()
        cooc.setdiag(0.0)
        # Reweight destinations so globally frequent tokens are less dominant.
        cooc = cooc.multiply(inv_token_counts[None, :]).tocsr()
        cooc.eliminate_zeros()

        # Prune each token row to top-k edges for sparsity.
        rows, cols, vals = [], [], []
        non_empty_rows = set()

        for i in range(n_tokens):
            start, end = cooc.indptr[i], cooc.indptr[i + 1]
            idxs = cooc.indices[start:end]
            ws = cooc.data[start:end]
            if idxs.size == 0:
                continue

            if idxs.size > self.top_k:
                # Keep only strongest neighbors for this row.
                pick = np.argpartition(ws, -self.top_k)[-self.top_k :]
                idxs = idxs[pick]
                ws = ws[pick]

            rows.extend([i] * int(idxs.size))
            cols.extend(idxs.tolist())
            vals.extend(ws.tolist())
            non_empty_rows.add(i)

        transition = csr_matrix((vals, (rows, cols)), shape=(n_tokens, n_tokens), dtype=float)
        # Empty rows get an explicit self-loop to keep rows normalizable.
        transition = self._add_self_loops_for_empty_rows(transition, n_tokens, non_empty_rows)
        # Final row L1 normalization enforces transition probabilities.
        return normalize(transition, norm="l1", axis=1, copy=False).tocsr()
