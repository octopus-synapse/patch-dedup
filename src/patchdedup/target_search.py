"""Bounded development-only search for simultaneous AP/F1 goals.

Cross-fitting only covers postprocessing: model epochs and repeated selection still
use development labels. These numbers must never be reported as an independent test.
"""

import itertools
import time

import numpy as np

from patchdedup.calibration import binary_threshold
from patchdedup.score_graph import crossfit_graph, fit_score_graph


def meets_targets(metrics, minimum_ap=0.99, minimum_f1=0.927):
    return bool(metrics["average_precision"] >= minimum_ap and metrics["f1"] >= minimum_f1)


def metric_key(metrics, minimum_f1=0.927):
    ap, f1 = metrics["average_precision"], metrics["f1"]
    feasible = f1 >= minimum_f1
    return (feasible, ap if feasible else f1, f1 if feasible else ap)


def search_scores(
    matrix, y, groups, minimum_ap=0.99, minimum_f1=0.927, max_ensembles=128, seed=42, deadline=None
):
    """Singles, pair mixtures, stacking and seeded mixtures; stop on both goals.

    Every evaluated F1 uses thresholds fitted on other groups. All rows count,
    including examples that a production review policy might abstain on.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or x.shape[1] == 0:
        raise ValueError("Expected nonempty score matrix aligned with development labels")
    if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Invalid probability matrix")
    outcomes = []
    best = None

    def evaluate(template):
        nonlocal best
        result = crossfit_graph(x, y, groups, template, threshold_metric="f1")
        result["fold_thresholds"] = [r["binary_threshold"] for r in result["fold_reports"]]
        outcomes.append(
            {k: v for k, v in result.items() if not k.startswith("oof") and k != "raw_oof_scores"}
        )
        if best is None or metric_key(result, minimum_f1) > metric_key(best, minimum_f1):
            best = result
        return meets_targets(best, minimum_ap, minimum_f1)

    def expired():
        return deadline is not None and time.monotonic() >= deadline

    hit = False
    for i in range(x.shape[1]):
        if expired() and best is not None:
            break
        hit = evaluate(dict(kind="average", indices=[i], weights=[1.0], calibrator_kind="identity"))
        if hit:
            break
    # Rank singles first, then explore all pairs in the top six score columns.
    ranked = sorted(outcomes, key=lambda r: metric_key(r, minimum_f1), reverse=True)
    pool = [r["template"]["indices"][0] for r in ranked[:6]]
    templates = []
    if not hit:
        for i in range(x.shape[1]):
            for calibrator in ("sigmoid", "isotonic"):
                templates.append(
                    dict(kind="average", indices=[i], weights=[1.0], calibrator_kind=calibrator)
                )
        for a, b in itertools.combinations(pool, 2):
            for w in (0.1, 0.25, 0.5, 0.75, 0.9):
                templates.append(
                    dict(
                        kind="average",
                        indices=[a, b],
                        weights=[w, 1 - w],
                        calibrator_kind="identity",
                    )
                )
        for count in range(2, len(pool) + 1):
            for c in (0.01, 0.1, 1.0, 10.0):
                templates.append(
                    dict(kind="stacking", indices=pool[:count], C=c, calibrator_kind="identity")
                )
        rng = np.random.default_rng(seed)
        for _ in range(max_ensembles):
            if len(pool) < 2:
                break
            templates.append(
                dict(
                    kind="average",
                    indices=pool,
                    weights=rng.dirichlet(np.ones(len(pool))).tolist(),
                    calibrator_kind="identity",
                )
            )
        for template in templates:
            if expired():
                break
            if evaluate(template):
                hit = True
                break
    graph = fit_score_graph(
        x, y, best["template"], calibration_training_scores=best["raw_oof_scores"]
    )
    graph["binary_threshold"] = binary_threshold(y, best["oof_scores"], metric="f1")
    return dict(
        target_met=hit,
        best=best,
        graph=graph,
        graphs_evaluated=len(outcomes),
        outcomes=outcomes,
        evaluation_split="development",
        evaluation_role="adaptive_development",
        independent_test_passed=False,
        independent_test=False,
        limitation="Repeated development selection is adaptive; requires a new independent test",
    )
