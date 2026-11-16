import importlib.util
import json
from pathlib import Path


def test_v03_render_preserves_curated_root_readme_and_archived_readme(tmp_path, monkeypatch):
    file = Path(__file__).resolve().parents[1] / "scripts/render_v03.py"
    spec = importlib.util.spec_from_file_location("render_v03", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for directory in ["reports/v03/candidates", "docs", "hub/dataset"]:
        (tmp_path / directory).mkdir(parents=True)
    metric = dict(
        average_precision=0.8,
        f1=0.7,
        macro_f1=0.7,
        coverage=0.5,
        selective_precision=0.8,
        conformal_marginal_coverage=0.9,
    )
    result = dict(metrics=metric, primary_pretrained_baseline=metric, test_pairs=20, test_groups=10)
    (tmp_path / "reports/v03/test.json").write_text(json.dumps(result))
    (tmp_path / "reports/v03/selection.json").write_text(
        json.dumps(dict(graphs_evaluated=1, selected={"macro_f1": 0.7}))
    )
    (tmp_path / "hub/dataset/README.md").write_text("config_name: v03\n")
    (tmp_path / "README.md").write_text(
        "# Curated overview\n\nASSIN and published v03 are distinct.\n"
    )
    (tmp_path / "docs/README-v02.md").write_text("# Historical overview\n")
    before = (tmp_path / "README.md").read_bytes()
    historical = (tmp_path / "docs/README-v02.md").read_bytes()
    monkeypatch.chdir(tmp_path)
    module.main()
    assert (tmp_path / "README.md").read_bytes() == before
    assert (tmp_path / "docs/README-v02.md").read_bytes() == historical
    assert "AP 0.800" in (tmp_path / "reports/v03/README.md").read_text()
    assert (tmp_path / "hub/mdeberta/README.md").exists()
