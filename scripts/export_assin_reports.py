"""Export portable ASSIN evidence without weights, dataset texts or runtime logs."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "external_metrics": "external-assin-ptbr/metrics.json",
    "external_protocol": "external-assin-ptbr/protocol.json",
    "finetune": "assin-finetune/comparison.json",
    "data_protocol": "assin-finetune/protocol.json",
    "refinement": "assin-refinement/comparison.json",
    "selection": "assin-refinement/selection.json",
    "protocol": "assin-refinement/protocol.json",
    "validation": "assin-refinement/validation-candidates.json",
    "graph": "assin-refinement/score-graph.json",
}
METRICS = (
    "support",
    "positive_support",
    "average_precision",
    "f1",
    "precision",
    "recall",
    "macro_f1",
    "confusion_matrix",
    "coverage",
    "review_rate",
)


def metrics(value):
    return {key: value[key] for key in METRICS if key in value}


def portable_component(root, component):
    directory = Path(component["directory"])
    absolute = directory.resolve() if directory.is_absolute() else (root / directory).resolve()
    if not absolute.is_relative_to(root):
        raise ValueError("Component directory is outside the repository")
    return dict(
        directory=absolute.relative_to(root).as_posix(),
        **{k: component[k] for k in ("model_sha256", "aggregation", "quantized") if k in component},
    )


def candidate(value):
    result = {k: value[k] for k in ("name", "weights", "policy", "indices") if k in value}
    result["metrics"] = metrics(value["metrics"])
    if "professional_development" in value:
        result["professional_development"] = metrics(value["professional_development"])
    return result


def export_reports(root, output):
    root, output = Path(root).resolve(), Path(output)
    sources, hashes = {}, {}
    for key, name in INPUTS.items():
        path = root / "artifacts" / name
        payload = path.read_bytes()
        sources[key] = json.loads(payload.decode("utf-8"))
        hashes["artifacts/" + name] = hashlib.sha256(payload).hexdigest()
    selection = sources["selection"]
    components = [portable_component(root, c) for c in selection["components"]]
    graph = dict(
        sources["graph"],
        components=[portable_component(root, c) for c in sources["graph"]["components"]],
    )
    if components != graph["components"]:
        raise ValueError("Score graph and selected components disagree")
    summary = dict(
        stages={
            "external": metrics(sources["external_metrics"]),
            "finetune": metrics(sources["finetune"]["after"]),
            "refinement": metrics(sources["refinement"]["after"]),
        },
        test_role="repeated_external_diagnostic",
        independent_test=False,
        positive="ASSIN PARAPHRASE (2); NONE (0) and ENTAILMENT (1) are negative",
    )
    selected = dict(
        selected=candidate(selection["selected"]),
        components=components,
        source_graph_sha256=selection["graph_sha256"],
        validation_pairs=selection["validation_pairs"],
        validation_positives=selection["validation_positives"],
        professional_development_pairs=selection["professional_development_pairs"],
        test_used_for_selection=selection["test_used_for_selection"],
    )
    external = sources["external_protocol"]
    data = sources["data_protocol"]
    protocol = dict(
        dataset={
            k: external[k]
            for k in (
                "dataset",
                "revision",
                "dataset_sha256",
                "subset",
                "split",
                "contamination",
                "license",
            )
            if k in external
        },
        external_evaluation={
            k: external[k]
            for k in ("graph_sha256", "binary_threshold", "positive", "negative", "domain")
            if k in external
        },
        prepared_data={
            k: data[k]
            for k in ("train_pairs", "validation_pairs", "test_pairs", "files", "source_sha256")
            if k in data
        },
        candidate_configs=sources["protocol"]["candidates"],
        code_sha256=sources["protocol"]["code_sha256"],
        artifact_sha256=hashes,
        paths_relative_to="repository root",
        local_weights_only=True,
        test_role="Known diagnostic; selection used validation only",
    )
    if "selection" in sources["finetune"]:
        first = sources["finetune"]["selection"]
        protocol["first_finetune_selection"] = {
            k: first[k]
            for k in ("weight_continued", "average_precision", "f1", "policy")
            if k in first
        }
        protocol["first_finetune_selection"]["components"] = [
            portable_component(root, c) for c in first["components"]
        ]
    training = []
    for config in protocol["candidate_configs"]:
        path = root / "artifacts/assin-refinement/candidates" / config["name"] / "run/training.json"
        if path.exists():
            payload = path.read_bytes()
            report = json.loads(payload.decode("utf-8"))
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(payload).hexdigest()
            training.append(
                dict(
                    name=config["name"],
                    **{
                        k: report[k]
                        for k in (
                            "config",
                            "selected_epoch",
                            "optimizer_steps",
                            "training_pairs",
                            "validation_pairs",
                            "trainable_parameters",
                            "total_parameters",
                            "initial_sha256",
                            "head_reset",
                            "sampling",
                            "positive_weight",
                        )
                        if k in report
                    },
                    packages=report.get("environment", {}).get("packages", {}),
                    code_commit=report.get("environment", {}).get("code_commit"),
                    source_sha256=report.get("environment", {}).get("source_sha256", {}),
                )
            )
    lines = [
        "# ASSIN: avaliação externa e refinamento local",
        "",
        "Comparação nos mesmos pares do teste pt-BR do ASSIN, com paráfrase como classe positiva.",
        "O teste foi consultado em rodadas anteriores; os ajustes foram selecionados na validação.",
        "",
        "| Etapa | AP | F1 | Precisão | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "external": "Pipeline antes de treinar no ASSIN",
        "finetune": "Primeiro fine-tuning",
        "refinement": "Refinamento com exemplos difíceis e ranking",
    }
    for stage, row in summary["stages"].items():
        lines.append(
            "| "
            + labels[stage]
            + " | "
            + " | ".join(
                f"{row[k]:.4f}" for k in ("average_precision", "f1", "precision", "recall")
            )
            + " |"
        )
    final = summary["stages"]["refinement"]
    lines += [
        "",
        f"O teste contém {final['support']} pares, dos quais {final['positive_support']} são positivos.",
        f"A configuração selecionada foi `{selected['selected']['name']}`.",
        f"A validação ASSIN usada na última rodada tem {selected['validation_pairs']} pares e "
        f"{selected['validation_positives']} positivos; os "
        f"{selected['professional_development_pairs']} pares profissionais são desenvolvimento reutilizado.",
        "",
        "## Evidências e reprodução",
        "",
        "- [Métricas](metrics.json): três etapas no mesmo teste, com matrizes de confusão.",
        "- [Seleção](selection.json): política, pesos, hashes e métricas de desenvolvimento.",
        "- [Grafo](score-graph.json): caminhos relativos à raiz do projeto; não é um bundle publicado.",
        "- [Validação](validation-candidates.json): candidatos e misturas avaliados antes do teste.",
        "- [Treinos](training.json): configurações, épocas selecionadas e versões das bibliotecas.",
        "- [Proveniência](protocol.json): revisão do dataset e hashes das fontes locais.",
        "",
        "Para regenerar este snapshot a partir dos artefatos locais:",
        "",
        "```bash",
        "uv run --locked python scripts/export_assin_reports.py",
        "```",
        "",
        "Os pesos refinados e os textos ASSIN permanecem locais. A ficha do dataset no snapshot",
        "registra licença desconhecida; este repositório não redistribui esse corpus nem atribui",
        "a ele a licença dos dados sintéticos do projeto. A exposição da base durante pré-treino",
        "não foi verificada. O modelo público continua sendo a v03 de configs/release.json.",
        "",
        "## Interpretação",
        "",
        "A queda em relação aos dados sintéticos mostra mudança de domínio e de distribuição.",
        "Notícias não validam redundância entre realizações profissionais. O teste repetido é",
        "diagnóstico e não substitui um novo teste independente com revisão humana.",
        "O limiar foi escolhido na validação; F1 usa classificação binária em todos os pares.",
        "Scores e a política de revisão não representam probabilidades calibradas.",
        "Nenhuma técnica deve ser declarada benéfica isoladamente apenas por participar da mistura.",
    ]
    exports = {
        "metrics.json": summary,
        "selection.json": selected,
        "protocol.json": protocol,
        "score-graph.json": graph,
        "validation-candidates.json": [candidate(c) for c in sources["validation"]],
        "training.json": training,
    }
    # Parse and validate all inputs before creating the export directory.
    output.mkdir(parents=True, exist_ok=True)
    for name, value in exports.items():
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/assin")
    args = parser.parse_args()
    export_reports(args.root, args.output)
