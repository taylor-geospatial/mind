import unittest

import numpy as np
import torch
from sklearn.metrics import r2_score

from mind import fit_ridge_cp
from scripts.paper.tikhonov_probe import build_profiles, path_scores


class RidgeTests(unittest.TestCase):
    def test_generalized_ridge_solution(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(30, 9)) * np.arange(1, 10) + 4
        features[:, 0] = 2
        targets = rng.normal(size=(30, 2)) + 10
        test = rng.normal(size=(5, 9)) + 12
        model = fit_ridge_cp(features, targets, alpha=2, beta=3, boundaries=(3, 6))
        mean, scale = features.mean(0), features.std(0, ddof=1).clip(min=1e-6)
        design = np.column_stack(((features - mean) / scale, np.ones(30)))
        penalty = np.diag([1, 1, 1, 3, 3, 3, 9, 9, 9, 0])
        weights = np.linalg.solve(design.T @ design + 2 * penalty, design.T @ targets)
        expected = np.column_stack(((test - mean) / scale, np.ones(5))) @ weights
        np.testing.assert_allclose(model.predict(test), expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(model.predict(test[:1]), expected[:1])

    def test_matches_paper_probe(self) -> None:
        rng = np.random.default_rng(4)
        features = rng.normal(size=(60, 140))
        targets = rng.normal(size=(60, 1)) + 5
        train, test = torch.arange(40), torch.arange(40, 60)
        profiles = build_profiles(140, widths=(), betas=(1, 10, 1000), pows=())
        scores = path_scores(
            torch.from_numpy(features),
            torch.from_numpy(targets),
            [("target", "regression", slice(0, 1), None)],
            train,
            test,
            profiles,
            (0.1, 10),
            torch.device("cpu"),
        )
        for i, beta in enumerate((1, 10, 1000)):
            for j, alpha in enumerate((0.1, 10)):
                with self.subTest(alpha=alpha, beta=beta):
                    model = fit_ridge_cp(features[:40], targets[:40, 0], alpha=alpha, beta=beta)
                    actual = r2_score(targets[40:, 0], model.predict(features[40:]))
                    np.testing.assert_allclose(actual, scores[i, j, 0], rtol=0, atol=1e-7)

    def test_invalid_inputs(self) -> None:
        features, targets = np.ones((4, 3)), np.ones(4)
        for kwargs in ({"alpha": 0}, {"beta": 0.5}, {"boundaries": (2, 1)}):
            with self.subTest(kwargs=kwargs), np.testing.assert_raises(ValueError):
                fit_ridge_cp(features, targets, **kwargs)
        with np.testing.assert_raises(ValueError):
            fit_ridge_cp(features, targets).predict(np.ones((2, 4)))
