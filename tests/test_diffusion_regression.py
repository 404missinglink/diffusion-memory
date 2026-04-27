import numpy as np

from lib.diffusion_memory import DiffusionMemory


def test_diffusion_regression_small_corpus():
    texts = ["a a a a", "a b c", "d e f g h"]
    memory = DiffusionMemory(texts)
    scored = memory.score_importance()

    result = memory.simulate_graph_diffusion_memory(
        scored["importance"],
        stimulus_steps=3,
        max_total_steps=10,
        stimulus_seed=1,
    )

    expected_importance = np.array(
        [
            0.0,
            0.33333333,
            0.99999999,
            0.33333333,
            0.33333333,
            0.33333333,
            0.33333333,
            0.33333333,
            0.33333333,
        ]
    )
    expected_scores = np.array(
        [
            0.20626243,
            0.23068617,
            0.22930419,
            0.20007849,
            0.23068617,
            0.20007849,
            0.23068617,
            0.23068617,
            0.23068617,
        ]
    )
    expected_alive_counts = np.array([2, 4, 9, 9, 9, 9, 9, 9, 9, 9])

    np.testing.assert_allclose(scored["importance"], expected_importance, atol=1e-8)
    np.testing.assert_allclose(result["scores"], expected_scores, atol=1e-8)
    np.testing.assert_array_equal(
        np.array([int(mask.sum()) for mask in result["alive_history"]]),
        expected_alive_counts,
    )
