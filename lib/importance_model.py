import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ImportanceModel:
    """Utility for converting token statistics into normalized importance scores."""

    @staticmethod
    def safe_unit_scale(values: np.ndarray) -> np.ndarray:
        # Stable [0, 1] scaling even for near-constant vectors.
        span = np.ptp(values)
        return (values - values.min()) / (span + 1e-8)

    @classmethod
    def score_from_arrays(cls, tf: np.ndarray, df: np.ndarray, doc_count: int) -> dict:
        # Combine frequency and IDF into a normalized importance score.
        n_tokens = int(tf.size)
        if n_tokens == 0:
            return {"importance": np.array([], dtype=float), "signals_norm": np.zeros((0, 2), dtype=float)}

        idf = np.log((1 + doc_count) / (1 + df)) + 1.0
        # Min-max normalize each signal before averaging.
        signals_norm = MinMaxScaler().fit_transform(np.column_stack([tf, idf]))
        importance = cls.safe_unit_scale(signals_norm.mean(axis=1))
        return {"importance": importance, "signals_norm": signals_norm}
