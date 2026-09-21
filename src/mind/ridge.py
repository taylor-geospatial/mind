"""Ridge regression with MIND's Chunked Penalty."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_array, check_X_y


@dataclass
class RidgeCP:
    """Fitted probe returned by ``fit_ridge_cp``; coefficients use standardized features."""

    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: float | np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict from embeddings using the training-set normalization."""
        features = check_array(features, dtype=np.float64)
        if features.shape[1] != self.mean_.size:
            raise ValueError(f"Expected {self.mean_.size} embedding dimensions")
        return ((features - self.mean_) / self.scale_) @ self.coef_.T + self.intercept_


def fit_ridge_cp(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 10.0,
    boundaries: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
) -> RidgeCP:
    """Fit a Chunked Penalty ridge probe to embeddings and regression targets.

    Features have shape [samples, dimensions]; targets may have one or several columns.
    Standardize with training means and sample standard deviations, floored at 1e-6.
    Chunk j has penalty alpha * beta**j, starting at j=0. The bias has penalty alpha,
    matching the paper's probes. beta=1 gives a uniform penalty across channels.
    Boundaries mark chunk starts after the first chunk; defaults match MIND.
    Select alpha and beta on validation data before fitting the final probe.
    """
    features, targets = check_X_y(
        features, targets, dtype=np.float64, multi_output=True, y_numeric=True, ensure_min_samples=2
    )
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not np.isfinite(beta) or beta < 1:
        raise ValueError("beta must be finite and at least 1")
    stops = np.asarray(boundaries)
    if stops.ndim != 1 or (
        stops.size
        and (
            not np.issubdtype(stops.dtype, np.integer)
            or np.any(stops <= 0)
            or np.any(np.diff(stops) <= 0)
        )
    ):
        raise ValueError("boundaries must be increasing positive integers")

    mean = features.mean(axis=0)
    scale = features.std(axis=0, ddof=1).clip(min=1e-6)
    chunk = np.searchsorted(stops, np.arange(features.shape[1]), side="right")
    rescale = float(beta) ** (-0.5 * chunk)
    design = np.column_stack(((features - mean) / scale * rescale, np.ones(len(features))))
    weights = Ridge(alpha=alpha, fit_intercept=False).fit(design, targets).coef_
    return RidgeCP(mean, scale, weights[..., :-1] * rescale, weights[..., -1])
