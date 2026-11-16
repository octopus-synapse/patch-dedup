"""Render factual reports and publication cards from completed experiment JSON."""

import json
from pathlib import Path


def load(name):
    return json.loads(Path(name).read_text(encoding="utf-8"))


def main():
    result = load("reports/v03/test.json")
    selection = load("reports/v03/selection.json")
    metric = result["metrics"]
    baseline = result["primary_pretrained_baseline"]
    candidates = [load(p) for p in sorted(Path("reports/v03/candidates").glob("*.json"))]
    lines = ["| Candidato | AP desenvolvimento | F1 desenvolvimento |", "|---|---:|---:|"]
    for c in candidates:
        lines.append(
            f"| {c['name']} | {c['validation']['average_precision']:.3f} | {c['validation']['f1']:.3f} |"
        )
    table = "\n".join(lines)
    report = f"""# PatchDedup v03 — transferência de NLI e seleção por grupos

Resultado da política selecionada no teste reservado de {result["test_pairs"]} pares
em {result["test_groups"]} cenários: **AP {metric["average_precision"]:.3f},
F1 {metric["f1"]:.3f}, macro-F1 {metric["macro_f1"]:.3f}**.
A base mDeBERTa de NLI sem ajuste teve AP {baseline["average_precision"]:.3f}
e F1 {baseline["f1"]:.3f} no mesmo teste, com política fixada no desenvolvimento.
Não confundir o ganho sobre embeddings antigos com o ganho sobre essa base forte.

A seleção de scores avaliou {selection["graphs_evaluated"]} grafos em três folds
por cenário. A macro-F1 cross-fitted selecionada foi {selection["selected"]["macro_f1"]:.3f}.
A seleção de épocas também usou desenvolvimento; cross-fitting não elimina todo
viés de seleção adaptativa. O teste só foi aberto após congelar modelo, grafo e calibração.

## Confiança e limitações

Cobertura fora de review: {metric["coverage"]:.3f}; precisão seletiva:
{metric["selective_precision"] if metric["selective_precision"] is not None else "sem decisões positivas"}.
A calibração reservada tem somente 20 pares de dez grupos. A cobertura marginal
conformal observada no teste foi {metric["conformal_marginal_coverage"]:.3f}; sua
interpretação requer exchangeability e não garante precisão ou proteção a domain shift.
Os dados foram escritos e revisados pela mesma IA, sem revisão humana ou CVs reais.
Resultados são de um piloto sintético, não evidência de prontidão para produção.

## Candidatos e técnicas

{table}

Foram avaliados transferência multilíngue de NLI, supervisão direcional multitarefa,
BCE ponderada, congelamento, learning rates por camada, AdamW, warmup/cosseno,
clipping/acumulação, BF16, checkpointing, label smoothing, R-Drop, EMA, LoRA,
distillation, pseudo-rótulos por consenso, mineração ponderada de negativos,
adaptação de linguagem mascarada, múltiplas seeds, HPO TPE/Hyperband, agregações,
calibração, ensembles/stacking, conformal e int8 CPU. Cada técnica tem sua evidência
nos JSONs; nenhuma deve ser declarada benéfica só por ter sido executada.
O queue de active learning aguarda revisão humana; não houve adjudicação humana.

A adaptação de linguagem usou um corpus sintético pequeno e uma cabeça MLM nova;
não foi pré-treino de larga escala. ASSIN2 não foi incorporado por licença não
confirmada no levantamento. RAG, DPO/RLHF, multimodalidade e MoE não pertencem à
tarefa discriminativa/dados disponíveis. Métodos de embeddings anteriores ficaram
preservados como baselines, sem trocar o objetivo principal de NLI.

## Reproduzir e inspecionar

`configs/experiment-v03.json` contém o protocolo. `selection.json`, `calibration.json`
e `test.json` preservam as escolhas e métricas; `test-predictions.json` e
`selected-errors.json` mostram as previsões reais. Pesos/checkpoints ficam fora
do Git; a publicação usa safetensors e manifesto. Execute a CLI com o bundle local
ou Docker CPU offline. Nenhum serviço de treino/inferência pago foi habilitado.
"""
    Path("reports/v03/README.md").write_text(report, encoding="utf-8")
    Path("docs/ARTICLE-v03.md").write_text(
        report + "\n\nRelato técnico de portfólio, sem revisão acadêmica por pares.\n",
        encoding="utf-8",
    )
    card = (
        """---
language:
- pt
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
datasets:
- enzoferracini/patchdedup-synthetic-ptbr
tags:
- patchdedup
- experimental
- nli
---
# PatchDedup PT-BR mDeBERTa — pipeline experimental

"""
        + report
        + """
## Carregamento e significado das saídas

Os pesos na raiz são do componente NLI principal. AutoModelForSequenceClassification
retorna logits NLI (entailment/neutral/contradiction), não a decisão final de
redundância. A CLI PatchDedup carrega score_graph.json, combina as orientações e
componentes, aplica calibração e retorna conjuntos conformais. O cosseno opcional
vem do encoder separado em embeddings/ e não é probabilidade de redundância.

Baixe a revisão fixa registrada em configs/release.json no GitHub e execute:
`patchdedup compare --model artifacts/model --text-a 'A' --text-b 'B'`.
A inferência é local, sem APIs. Scores não verificam fatos. Não apagar bullets,
avaliar candidatos ou classificar competência. O uso validado é somente pesquisa
neste conjunto sintético em português. Não há inglês validado ou impacto medido.

Pesos NLI/derivados: MIT, com atribuição a Moritz Laurer e autores das bases.
O encoder de embeddings separado deriva de Sentence Transformers sob Apache-2.0.
Dados: CC BY 4.0. Código: MIT. Preservar LICENSE, NOTICE e os cards de origem.
Autoria do projeto: Enzo Ferracini Patti com Codex, sem afiliação ao iFood.
Código e evidências: https://github.com/octopus-synapse/patch-dedup
"""
    )
    Path("hub/mdeberta").mkdir(parents=True, exist_ok=True)
    Path("hub/mdeberta/README.md").write_text(card, encoding="utf-8")
    dataset = Path("hub/dataset/README.md").read_text(encoding="utf-8")
    if "config_name: v03" not in dataset:
        addition = """configs:
- config_name: v03
  default: true
  data_files:
  - split: train
    path: v03/train.jsonl
  - split: validation
    path: v03/validation.jsonl
  - split: calibration
    path: v03/calibration.jsonl
  - split: test
    path: v03/test.jsonl
"""
        dataset = (
            dataset.replace("  default: true\n", "")
            .replace("configs:\n", addition, 1)
            .replace("- n<1K", "- 1K<n<10K")
        )
        dataset += """\n## Bateria v03\n\nA configuração v03 inclui 744 pares de treino, 132 de desenvolvimento, 20 de\ncalibração e 40 de teste novo. Os 248 pares adicionais de treino são combinações\ndos mesmos cenários; nenhum cenário de avaliação foi incorporado ao treino.\nPseudo-rótulos, quando presentes, estão em v03/pseudo_train.jsonl, não fazem parte\ndo treino padrão e não foram revisados por humanos. A coluna split dos registros\nde calibração conserva validation por compatibilidade; o arquivo/manifesto\ndistingue seu papel reservado. Os dados de todas as versões são sintéticos.\n"""
    Path("hub/dataset/README.md").write_text(dataset, encoding="utf-8")
    old = Path("README.md").read_text(encoding="utf-8")
    marker = "<!-- V03 AUTO SUMMARY -->"
    if marker not in old:
        Path("docs/README-v02.md").write_text(old, encoding="utf-8")
    root = f"""# PatchDedup

{marker}
Pipeline experimental em português para redundância entre realizações profissionais.
No teste reservado v03: **AP {metric["average_precision"]:.3f}, F1 {metric["f1"]:.3f}**.
A base mDeBERTa NLI sem ajuste teve AP {baseline["average_precision"]:.3f} e
F1 {baseline["f1"]:.3f}. Dados sintéticos, sem revisão humana: resultados ainda
não demonstram qualidade em currículos reais. Não excluir bullets automaticamente.

[Modelo mDeBERTa](https://huggingface.co/enzoferracini/patchdedup-ptbr-mdeberta)
· [Dataset](https://huggingface.co/datasets/enzoferracini/patchdedup-synthetic-ptbr)
· [Relatório atual](reports/v03/README.md)
· [Artigo](docs/ARTICLE-v03.md)

## Executar localmente

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu hf download enzoferracini/patchdedup-ptbr-mdeberta --local-dir artifacts/model
uv run --locked --extra cpu patchdedup compare --model artifacts/model --text-a 'A' --text-b 'B'
```

Use o SHA fixo de configs/release.json para reprodução. O download requer rede;
a inferência é local. A raiz contém o componente NLI principal; embeddings/
contém o encoder separado de cosseno. Use a CLI para executar todo o grafo,
calibração e conjuntos conformais, não somente os logits NLI de uma orientação.

## Pesquisa e engenharia

[Protocolo](configs/experiment-v03.json), [métodos](docs/TECHNIQUES.md),
[contratos](docs/CONTRACTS.md) e [TDD](CONTRIBUTING.md). As versões
[v01](reports/v01/README.md) e [v02](reports/v02/README.md) preservam resultados
negativos, inclusive a falha de generalização dos primeiros bi-encoders.

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv build
bash scripts/docker_smoke.sh artifacts/model
```

Docker CPU, não root, modelo read-only e inferência offline. A CI remota permanece
sujeita ao bloqueio de faturamento da conta; nenhum orçamento foi aumentado.
Código MIT; modelos NLI MIT, embeddings Apache-2.0 e dados CC BY 4.0.
Projeto independente, sem afiliação ao iFood.

## Executor autônomo

```bash
bash scripts/finish_remaining.sh --background --publish
```

Retoma etapas pendentes, gera relatórios e publica somente após verificações.
Computador ligado e sem suspender. Logs/status em artifacts/v03/orchestration/.
"""
    Path("README.md").write_text(root, encoding="utf-8")


if __name__ == "__main__":
    main()
