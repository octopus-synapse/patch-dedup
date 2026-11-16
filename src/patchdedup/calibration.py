"""Cross-fitted calibration and split conformal sets, with explicit assumptions."""

import math

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


def group_folds(y, groups):
    y, groups = np.asarray(y), np.asarray(groups)
    return list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
            np.zeros((len(y), 1)), y, groups
        )
    )


def logit(scores):
    scores = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(scores / (1 - scores))


def fit_calibrator(kind, scores, y):
    s, y = np.asarray(scores, dtype=float), np.asarray(y)
    if kind == "identity":
        return dict(kind=kind)
    if kind == "sigmoid":
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(logit(s)[:, None], y)
        return dict(kind=kind, slope=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]))
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(s, y)
        return dict(kind=kind, x=model.X_thresholds_.tolist(), y=model.y_thresholds_.tolist())
    raise ValueError("Unknown calibrator")


def apply_calibrator(calibration, scores):
    s = np.asarray(scores, dtype=float)
    if calibration["kind"] == "identity":
        return s
    if calibration["kind"] == "sigmoid":
        z = calibration["slope"] * logit(s) + calibration["intercept"]
        return 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    if calibration["kind"] == "isotonic":
        return np.interp(s, calibration["x"], calibration["y"])
    raise ValueError("Unknown calibrator")


def binary_threshold(y, scores):
    """Choose macro F1 cutoff on fitting folds, never their held-out groups."""
    y, s = np.asarray(y), np.asarray(scores)
    choices = np.unique(np.concatenate(([0.0, 1.0], s)))
    pred = s[:, None] >= choices
    tp = (pred & (y[:, None] == 1)).sum(0)
    tn = (~pred & (y[:, None] == 0)).sum(0)
    fp = (pred & (y[:, None] == 0)).sum(0)
    fn = (~pred & (y[:, None] == 1)).sum(0)
    positive = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    negative = 2 * tn / np.maximum(2 * tn + fp + fn, 1)
    index = max(
        range(len(choices)),
        key=lambda i: (
            float((positive[i] + negative[i]) / 2),
            float(positive[i]),
            -abs(choices[i] - 0.5),
        ),
    )
    return float(choices[index])


def prediction_sets(scores, q):
    return [
        [label for label, error in ((0, float(p)), (1, 1 - float(p))) if error <= q] for p in scores
    ]


def conformal_policy(y, scores, alpha=0.1):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    errors = np.where(y == 1, 1 - scores, scores)
    rank = math.ceil((len(y) + 1) * (1 - alpha))
    q = float(np.sort(errors)[rank - 1]) if rank <= len(y) else 1.0
    if q >= 1:
        thresholds = dict(low=0.0, high=1.0, positive_enabled=False, negative_enabled=False)
    elif q >= 0.5:
        thresholds = dict(
            low=1 - q,
            high=float(np.nextafter(q, 1.0)),
            positive_enabled=True,
            negative_enabled=True,
        )
    else:
        thresholds = dict(
            low=float(np.nextafter(q, 1.0)),
            high=1 - q,
            positive_enabled=True,
            negative_enabled=True,
        )
    return dict(
        alpha=alpha,
        q=q,
        calibration_pairs=len(y),
        thresholds=thresholds,
        assumption="Marginal coverage requires exchangeability; not a precision guarantee or domain-shift protection",
    )


def calibration_metrics(y, scores):
    y, p = np.asarray(y), np.clip(np.asarray(scores), 1e-7, 1 - 1e-7)
    ece = 0.0
    bins = np.minimum((p * 10).astype(int), 9)
    for b in range(10):
        selected = bins == b
        if selected.any():
            ece += selected.mean() * abs(y[selected].mean() - p[selected].mean())
    return dict(
        brier=float(np.mean((y - p) ** 2)),
        log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        expected_calibration_error_10_bins=float(ece),
    )
