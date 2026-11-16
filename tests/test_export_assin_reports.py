"""Software fixtures validate report export; these are not ML benchmark results."""

import importlib.util
import json
from pathlib import Path

import pytest


def exporter():
    file = Path(__file__).resolve().parents[1] / "scripts/export_assin_reports.py"
    spec = importlib.util.spec_from_file_location("export_assin_reports", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_artifacts(root):
    metrics = dict(
        average_precision=0.8,
        f1=0.7,
        precision=0.75,
        recall=0.65,
        support=20,
        positive_support=4,
        confusion_matrix=[[15, 1], [1, 3]],
    )
    component = dict(
        directory=str(root / "artifacts/local-model"),
        model_sha256="a" * 64,
        aggregation="mean",
        quantized=False,
    )
    chosen = dict(
        name="chosen",
        metrics=metrics,
        professional_development=metrics,
        policy={"binary_threshold": 0.4},
        weights=[1.0],
    )
    selection = dict(
        selected=chosen,
        components=[component],
        graph_sha256="b" * 64,
        validation_pairs=20,
        validation_positives=4,
        professional_development_pairs=20,
        test_used_for_selection=False,
    )
    values = {
        "external-assin-ptbr/metrics.json": dict(metrics, elapsed_seconds=10),
        "external-assin-ptbr/protocol.json": dict(
            dataset="nilc-nlp/assin",
            revision="c" * 40,
            dataset_sha256="d" * 64,
            subset="ptbr",
            split="test",
            contamination="Base pretraining exposure not verified",
            license="unknown",
        ),
        "assin-finetune/comparison.json": dict(before=metrics, after=metrics),
        "assin-finetune/protocol.json": dict(
            train_pairs=30,
            validation_pairs=20,
            test_pairs=20,
            files={"train.jsonl": "e" * 64, "validation.jsonl": "f" * 64, "test.jsonl": "0" * 64},
            source_sha256={"train": "1" * 64, "validation": "2" * 64},
        ),
        "assin-refinement/comparison.json": dict(
            before=metrics, after=metrics, changes={"average_precision": 0.0}, selection=selection
        ),
        "assin-refinement/selection.json": selection,
        "assin-refinement/protocol.json": dict(
            candidates=[dict(name="candidate", training={"epochs": 1})],
            code_sha256={"trainer.py": "3" * 64},
            training_pairs=30,
            validation_pairs=20,
            professional_development_pairs=20,
        ),
        "assin-refinement/validation-candidates.json": [chosen],
        "assin-refinement/score-graph.json": dict(components=[component], weights=[1.0]),
    }
    for name, value in values.items():
        p = root / "artifacts" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value), encoding="utf-8")
    return values


def test_export_preserves_metrics_hashes_and_portable_paths(tmp_path):
    module = exporter()
    fixture_artifacts(tmp_path)
    output = tmp_path / "reports/assin"
    module.export_reports(tmp_path, output)
    summary = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert summary["stages"]["refinement"]["average_precision"] == 0.8
    assert summary["test_role"] == "repeated_external_diagnostic"
    assert summary["independent_test"] is False
    selected = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert selected["components"][0]["directory"] == "artifacts/local-model"
    assert selected["components"][0]["model_sha256"] == "a" * 64
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["dataset"]["revision"] == "c" * 40
    assert all(len(digest) == 64 for digest in protocol["artifact_sha256"].values())
    content = "".join(p.read_text(encoding="utf-8") for p in output.iterdir())
    assert str(tmp_path) not in content
    assert "elapsed_seconds" not in content


def test_export_rejects_component_outside_repository_before_writing(tmp_path):
    module = exporter()
    values = fixture_artifacts(tmp_path)
    selected = values["assin-refinement/selection.json"]
    selected["components"][0]["directory"] = "/outside/private/model"
    (tmp_path / "artifacts/assin-refinement/selection.json").write_text(json.dumps(selected))
    output = tmp_path / "reports/assin"
    with pytest.raises(ValueError, match="outside"):
        module.export_reports(tmp_path, output)
    assert not output.exists()


def test_export_is_deterministic_and_does_not_copy_runtime_files(tmp_path):
    module = exporter()
    fixture_artifacts(tmp_path)
    (tmp_path / "artifacts/assin-refinement/status.json").write_text('{"pid":123}')
    (tmp_path / "artifacts/assin-refinement/predictions.json").write_text("private text")
    output = tmp_path / "reports/assin"
    module.export_reports(tmp_path, output)
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    module.export_reports(tmp_path, output)
    assert original == {p.name: p.read_bytes() for p in output.iterdir()}
    assert not (output / "status.json").exists()
    assert not (output / "predictions.json").exists()
