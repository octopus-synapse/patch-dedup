"""Post-hoc group uncertainty from saved predictions; never reruns selection."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from patchdedup.dataset import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, default=Path("reports/v01"))
    args = parser.parse_args()
    report = json.loads((args.reports / "test.json").read_text(encoding="utf-8"))
    names = ["base", report["chosen_finetuned"]]
    payloads = {
        name: json.loads(
            (args.reports / f"{name}-test-predictions.json").read_text(encoding="utf-8")
        )
        for name in names
    }
    reference = payloads["base"]["predictions"]
    group_ids = sorted({p["scenario_id"] for p in reference})
    grouped = {
        group: [i for i, p in enumerate(reference) if p["scenario_id"] == group]
        for group in group_ids
    }
    scores = {
        name: np.asarray([p["score"] for p in payloads[name]["predictions"]]) for name in names
    }
    y = np.asarray([p["label"] for p in reference])
    random = np.random.default_rng(42)
    distributions = {name: {"f1": [], "average_precision": []} for name in names}
    for _ in range(2000):
        sample = random.choice(group_ids, size=len(group_ids), replace=True)
        index = np.asarray([i for group in sample for i in grouped[group]])
        for name in names:
            s = scores[name][index]
            threshold = payloads[name]["policy"]["binary_threshold"]
            distributions[name]["f1"].append(float(f1_score(y[index], s >= threshold)))
            distributions[name]["average_precision"].append(
                float(average_precision_score(y[index], s))
            )
    intervals = {
        name: {
            metric: np.quantile(values, [0.025, 0.975]).tolist()
            for metric, values in metrics.items()
        }
        for name, metrics in distributions.items()
    }
    differences = {
        metric: np.quantile(
            np.asarray(distributions[names[1]][metric]) - np.asarray(distributions["base"][metric]),
            [0.025, 0.975],
        ).tolist()
        for metric in ("f1", "average_precision")
    }
    save_json(
        args.reports / "uncertainty.json",
        dict(
            seed=42,
            repetitions=2000,
            resampling_unit="scenario_id",
            groups=len(group_ids),
            percentile_95=intervals,
            selected_minus_base_95=differences,
            limitation="Descriptive pilot uncertainty; does not measure template/author/domain shift",
        ),
    )


if __name__ == "__main__":
    main()
