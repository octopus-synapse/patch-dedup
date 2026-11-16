"""JSON score graphs with group cross-fitting for calibration and ensemble selection."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from patchdedup.calibration import apply_calibrator, binary_threshold, fit_calibrator, group_folds


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -60, 60)))


def fit_combiner(matrix, y, template):
    x = np.asarray(matrix, dtype=float)[:, template["indices"]]
    graph = dict(template)
    if template["kind"] == "stacking":
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            C=template.get("C", 1.0), class_weight="balanced", max_iter=2000, random_state=42
        ).fit(scaler.transform(x), y)
        graph["meta"] = dict(
            mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(),
            coefficients=model.coef_[0].tolist(),
            intercept=float(model.intercept_[0]),
        )
    else:
        weights = np.asarray(template["weights"], dtype=float)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Ensemble weights must be nonnegative with positive sum")
        graph["weights"] = (weights / weights.sum()).tolist()
    return graph


def combine_scores(graph, matrix):
    x = np.asarray(matrix, dtype=float)[:, graph["indices"]]
    if graph["kind"] == "stacking":
        meta = graph["meta"]
        z = (x - np.asarray(meta["mean"])) / np.asarray(meta["scale"])
        return sigmoid(z @ np.asarray(meta["coefficients"]) + meta["intercept"])
    return np.clip(x @ np.asarray(graph["weights"]), 0, 1)


def fit_score_graph(matrix, y, template, calibration_training_scores=None):
    graph = fit_combiner(matrix, y, template)
    raw = combine_scores(graph, matrix)
    graph["calibrator"] = fit_calibrator(
        template["calibrator_kind"],
        raw if calibration_training_scores is None else calibration_training_scores,
        y,
    )
    return graph


def apply_score_graph(graph, matrix):
    return apply_calibrator(graph["calibrator"], combine_scores(graph, matrix))


def crossfit_graph(matrix, y, groups, template):
    x, y = np.asarray(matrix, dtype=float), np.asarray(y)
    oof = np.zeros(len(y))
    raw_oof = np.zeros(len(y))
    binary = np.zeros(len(y), dtype=int)
    folds = group_folds(y, groups)
    fold_reports = []
    for train, valid in folds:
        graph = fit_score_graph(x[train], y[train], template)
        p = apply_score_graph(graph, x[valid])
        cutoff = binary_threshold(y[train], apply_score_graph(graph, x[train]))
        oof[valid] = p
        raw_oof[valid] = combine_scores(graph, x[valid])
        binary[valid] = (p >= cutoff).astype(int)
        fold_reports.append(
            dict(
                train_groups=len(set(np.asarray(groups)[train])),
                validation_groups=len(set(np.asarray(groups)[valid])),
                macro_f1=float(f1_score(y[valid], binary[valid], average="macro")),
                average_precision=float(average_precision_score(y[valid], p)),
            )
        )
    return dict(
        macro_f1=float(f1_score(y, binary, average="macro")),
        f1=float(f1_score(y, binary)),
        average_precision=float(average_precision_score(y, oof)),
        folds=len(folds),
        fold_reports=fold_reports,
        oof_scores=oof.tolist(),
        raw_oof_scores=raw_oof.tolist(),
        oof_binary=binary.tolist(),
        template=template,
    )
