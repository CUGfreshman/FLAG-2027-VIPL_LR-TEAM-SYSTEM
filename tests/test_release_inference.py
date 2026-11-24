from pathlib import Path
import unittest

import numpy as np
import torch

from flag2027.inference import FEATURE_DIMS, score_features
from flag2027.submission import OUTPUTS


class ReleasedSystemTest(unittest.TestCase):
    def test_actual_checkpoints_score_each_protocol(self):
        root = Path(__file__).resolve().parents[1]
        rows = [{'subset': subset, 'trial_id': str(i)} for i, subset in enumerate(OUTPUTS)]
        rng = np.random.default_rng(2027)
        inputs = {key: rng.standard_normal((len(rows), width), dtype=np.float32)
                  for key, width in FEATURE_DIMS.items()}
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            scores, parts = score_features(inputs, rows, root / 'configs/system.json')
            self.assertEqual(parts.shape, (len(rows), 6))
            self.assertEqual(scores.shape, (len(rows),))
            self.assertTrue(np.isfinite(parts).all())
            self.assertTrue(np.all((parts >= 0) & (parts <= 2)))
            np.testing.assert_array_equal(scores, np.sum(parts, axis=1) / 6)
            for i, row in enumerate(rows):
                _, single = score_features({k: v[i:i+1] for k, v in inputs.items()},
                                           [row], root / 'configs/system.json')
                np.testing.assert_allclose(parts[i], single[0], rtol=0, atol=1e-5)
        finally:
            torch.set_num_threads(threads)
