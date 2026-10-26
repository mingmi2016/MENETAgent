"""Run without pytest: python -m unittest discover -s tests -p test_gblup.py."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from utils.gblup import GBLUPRegressor


class GBLUPTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        self.x = rng.choice([-1.0, 0.0, 1.0], size=(24, 32))
        self.y = self.x[:, :5].sum(axis=1) + rng.normal(0, 1, 24)
        self.new_x = rng.choice([-1.0, 0.0, 1.0], size=(6, 32))

    def test_matches_dense_mixed_model_and_reml(self):
        model = GBLUPRegressor().fit(self.x, self.y)
        z = self.x - self.x.mean(axis=0)
        p = (self.x.mean(axis=0) + 1) / 2
        scale = 2 * np.sum(p * (1 - p))
        k = z @ z.T / scale
        one = np.ones(len(self.y))
        def dense_reml(log_delta):
            v = k + np.exp(log_delta) * np.eye(len(one))
            inv_one = np.linalg.solve(v, one)
            inv_y = np.linalg.solve(v, self.y)
            mu = one @ inv_y / (one @ inv_one)
            residual = self.y - mu
            q = residual @ np.linalg.solve(v, residual)
            return ((len(one) - 1) * np.log(q / (len(one) - 1))
                    + np.linalg.slogdet(v)[1] + np.log(one @ inv_one))
        reference = minimize_scalar(dense_reml, bounds=(-20, 20), method="bounded")
        self.assertAlmostEqual(np.log(model.delta_), reference.x, places=3)
        alpha = np.linalg.solve(k + model.delta_ * np.eye(len(one)), self.y - self.y.mean())
        cross = (self.new_x - self.x.mean(axis=0)) @ z.T / scale
        np.testing.assert_allclose(model.predict(self.new_x), self.y.mean() + cross @ alpha, rtol=1e-7, atol=1e-7)

    def test_offset_allele_flip_and_reload(self):
        original = GBLUPRegressor().fit(self.x, self.y)
        shifted = GBLUPRegressor().fit(self.x, self.y + 100)
        flipped = GBLUPRegressor().fit(-self.x, self.y)
        np.testing.assert_allclose(shifted.predict(self.new_x), original.predict(self.new_x) + 100, atol=1e-6)
        np.testing.assert_allclose(flipped.predict(-self.new_x), original.predict(self.new_x), atol=1e-6)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "gblup.joblib"
            joblib.dump(original, path)
            np.testing.assert_allclose(joblib.load(path).predict(self.new_x), original.predict(self.new_x))

    def test_limits_and_invalid_inputs(self):
        for x, y in ((np.ones((24, 32)), self.y), (self.x * 3, self.y),
                     (self.x, np.full(24, np.nan))):
            with self.assertRaises(ValueError):
                GBLUPRegressor().fit(x, y)
        with self.assertRaises(MemoryError):
            GBLUPRegressor(max_memory_bytes=1).fit(self.x, self.y)
        model = GBLUPRegressor().fit(self.x, np.ones(24) * 7)
        np.testing.assert_allclose(model.predict(self.new_x), 7)
        with self.assertRaises(ValueError):
            model.predict(self.new_x[:, :-1])

    def test_monomorphic_markers_do_not_affect_predictions(self):
        original = GBLUPRegressor().fit(self.x, self.y)
        extended = GBLUPRegressor().fit(np.column_stack((self.x, np.ones(24))), self.y)
        np.testing.assert_allclose(extended.predict(np.column_stack((self.new_x, np.ones(6)))),
                                   original.predict(self.new_x), atol=1e-6)

    def test_training_integration_and_test_label_isolation(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from utils.train import evaluate_gblup_baseline
        def loader(x, y):
            return DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32),
                                           torch.zeros(len(x), 1), torch.tensor(y, dtype=torch.float32)))
        train = loader(self.x, self.y)
        test = loader(self.new_x, np.arange(6))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "gblup_model.joblib"
            metrics = evaluate_gblup_baseline(train, test, path)
            changed = evaluate_gblup_baseline(train, loader(self.new_x, np.arange(6) + 100))
            self.assertEqual(metrics["delta"], changed["delta"])
            self.assertEqual(metrics["train_sample_count"], 24)
            self.assertEqual(metrics["test_sample_count"], 6)
            for field in ("test_r2", "test_pearson_r", "test_rmse", "test_mae"):
                self.assertIn(field, metrics)
            self.assertEqual(len(joblib.load(path).predict(self.new_x)), 6)

    def test_one_baseline_failure_does_not_hide_other_methods(self):
        from utils.train import evaluate_baselines
        with patch("utils.train.evaluate_gblup_baseline", side_effect=MemoryError("limit")), \
             patch("utils.train.evaluate_ridge_baseline", return_value={"test_r2": .3}), \
             patch("utils.train.evaluate_random_forest_baseline", return_value={"test_r2": .4}), \
             patch("utils.train.evaluate_xgboost_baseline", return_value={"test_r2": .5}):
            metrics = evaluate_baselines(None, None)
        self.assertEqual(metrics["gblup"]["status"], "unavailable")
        self.assertEqual(metrics["xgboost"]["test_r2"], .5)
        self.assertIn("fit_and_test_seconds", metrics["genomic_ridge"])


if __name__ == "__main__":
    unittest.main()
