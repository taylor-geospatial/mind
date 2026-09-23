"""Keep the public and evaluation probes consistent with an unpenalized intercept."""

import unittest
import warnings

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score

from mind import fit_ridge_cp
from mind.eval.probe import (
    RIDGE_ALPHAS,
    _cv_best_alpha,
    _ridge_eval,
    _ridge_eval_alpha_path,
    _standardize_train_test,
    ridge_fold_predictions,
    torch_probe_score,
)
from scripts.paper.tikhonov_probe import path_scores, run_group

DEV = torch.device("cpu")


class ProbeInterceptTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 8)) * np.arange(1, 9) + 5
        signal = x[:, 0] - 2 * x[:, 1] + rng.normal(size=120)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(signal + 300, dtype=torch.float32)[:, None]
        self.classes = torch.tensor(np.digitize(signal, np.quantile(signal, [0.25, 0.6])))
        self.onehot = torch.nn.functional.one_hot(self.classes).float()
        self.train, self.test = torch.arange(90), torch.arange(90, 120)

    def test_direct_and_fast_paths_match_sklearn(self) -> None:
        for task, y, classes in (
            ("regression", self.y, None),
            ("classification", self.onehot, self.classes),
        ):
            for standardize in (False, True):
                xtr, xte = self.x[self.train], self.x[self.test]
                if standardize:
                    xtr, xte = _standardize_train_test(xtr, xte)
                fast = _ridge_eval_alpha_path(
                    self.x,
                    y,
                    classes,
                    self.train,
                    self.test,
                    RIDGE_ALPHAS,
                    task,
                    DEV,
                    standardize,
                )
                for i, alpha in enumerate(RIDGE_ALPHAS):
                    with self.subTest(task=task, standardize=standardize, alpha=alpha):
                        ref = Ridge(alpha=alpha).fit(xtr.double().numpy(), y[self.train].numpy())
                        pred = ref.predict(xte.double().numpy())
                        expected = (
                            r2_score(y[self.test].numpy(), pred)
                            if task == "regression"
                            else accuracy_score(self.classes[self.test].numpy(), pred.argmax(1))
                        )
                        direct = _ridge_eval(
                            self.x,
                            y,
                            classes,
                            self.train,
                            self.test,
                            alpha,
                            task,
                            DEV,
                            standardize,
                        )
                        np.testing.assert_allclose(direct, expected, rtol=0, atol=1e-6)
                        np.testing.assert_allclose(fast[i], expected, rtol=0, atol=1e-6)

    def test_chunked_paths_match_sklearn(self) -> None:
        xtr, xte = self.x[self.train], self.x[self.test]
        mean = xtr.mean(0)
        std = xtr.std(0).clamp_min(1e-6)
        xtr, xte = ((xtr - mean) / std).double().numpy(), ((xte - mean) / std).double().numpy()
        profiles = [("trunc", 4), ("chunk", np.repeat([1.0, 10.0], 4))]
        alphas = (0.01, 1.0, 100.0, 10000.0)
        for task, y, classes in (
            ("regression", self.y, None),
            ("classification", self.onehot, self.classes),
        ):
            scores = path_scores(
                self.x,
                y,
                [("target", task, slice(0, y.shape[1]), classes)],
                self.train,
                self.test,
                profiles,
                alphas,
                DEV,
            )
            for pi, (name, spec) in enumerate(profiles):
                a, b = (
                    (xtr[:, :spec], xte[:, :spec])
                    if isinstance(spec, int)
                    else (xtr * spec**-0.5, xte * spec**-0.5)
                )
                for ai, alpha in enumerate(alphas):
                    with self.subTest(task=task, profile=name, alpha=alpha):
                        pred = Ridge(alpha=alpha).fit(a, y[self.train].numpy()).predict(b)
                        expected = (
                            r2_score(y[self.test].numpy(), pred)
                            if task == "regression"
                            else accuracy_score(self.classes[self.test].numpy(), pred.argmax(1))
                        )
                        np.testing.assert_allclose(scores[pi, ai, 0], expected, rtol=0, atol=1e-6)

    def test_public_probe_matches_sklearn_and_preserves_target_shift(self) -> None:
        x, y = self.x.double().numpy(), self.y.double().numpy()
        mean, std = x[:90].mean(0), x[:90].std(0, ddof=1)
        for beta in (1.0, 10.0):
            for alpha in (1.0, 10000.0):
                with self.subTest(alpha=alpha, beta=beta):
                    scale = beta ** (-0.5 * np.repeat([0, 1], 4))
                    ref = Ridge(alpha=alpha).fit((x[:90] - mean) / std * scale, y[:90])
                    model = fit_ridge_cp(x[:90], y[:90], alpha=alpha, beta=beta, boundaries=(4,))
                    shifted = fit_ridge_cp(
                        x[:90], y[:90] + 1000, alpha=alpha, beta=beta, boundaries=(4,)
                    )
                    np.testing.assert_allclose(
                        model.predict(x[90:]), ref.predict((x[90:] - mean) / std * scale), atol=1e-9
                    )
                    np.testing.assert_allclose(
                        shifted.predict(x[90:]), model.predict(x[90:]) + 1000
                    )

    def test_cv_selection_uses_corrected_solver(self) -> None:
        folds = [torch.arange(i, 90, 3) for i in range(3)]
        alphas = (0.01, 1.0, 100.0, 10000.0)
        expected = []
        for alpha in alphas:
            scores = []
            for i, test in enumerate(folds):
                train = torch.cat([fold for j, fold in enumerate(folds) if j != i])
                xtr, xte = _standardize_train_test(self.x[train], self.x[test])
                pred = (
                    Ridge(alpha=alpha)
                    .fit(xtr.double().numpy(), self.y[train].numpy())
                    .predict(xte.double().numpy())
                )
                scores.append(r2_score(self.y[test].numpy(), pred))
            expected.append(np.mean(scores))
        for fast in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                alpha, score = _cv_best_alpha(
                    self.x,
                    self.y,
                    None,
                    self.train,
                    3,
                    alphas,
                    "regression",
                    DEV,
                    0,
                    fold_index_list=folds,
                    fast_alpha_path=fast,
                )
            assert alpha == alphas[int(np.argmax(expected))]
            np.testing.assert_allclose(score, max(expected), rtol=0, atol=1e-6)

    def test_fold_predictions_preserve_target_shift(self) -> None:
        x, y = self.x.numpy(), self.y.numpy().ravel()
        kwargs = {"task_type": "regression", "folds": 3, "device": "cpu", "alphas": (10000.0,)}
        base = ridge_fold_predictions(x, y, **kwargs)
        shifted = ridge_fold_predictions(x, y + 300, **kwargs)
        np.testing.assert_allclose(shifted["pred"], base["pred"] + 300, atol=1e-4, rtol=0)
        np.testing.assert_array_equal(base["row"], shifted["row"])
        np.testing.assert_array_equal(base["fold"], shifted["fold"])

    def test_spatial_nested_chunk_selection_matches_ridge(self) -> None:
        folds = [torch.arange(i, len(self.x), 4) for i in range(4)]
        fold_assign = np.arange(len(self.x)) % 4
        alphas = (0.1, 10.0, 1000.0)
        for task, y, classes in (
            ("regression", self.y, None),
            ("classification", self.onehot, self.classes),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scores, _ = run_group(
                    self.x,
                    y,
                    [("target", task, slice(0, y.shape[1]), classes)],
                    folds,
                    [("ridge_w8", 8)],
                    {},
                    DEV,
                    alphas=alphas,
                    folds=4,
                    spatial_inner=True,
                )
                ref = torch_probe_score(
                    self.x.numpy(),
                    self.y.numpy().ravel() if classes is None else classes.numpy(),
                    task,
                    folds=4,
                    device="cpu",
                    alphas=alphas,
                    fold_assign=fold_assign,
                    fast_alpha_path=True,
                )
            np.testing.assert_allclose(
                scores[("ridge_w8", "nested_spatial")][0], ref, rtol=0, atol=1e-6
            )
