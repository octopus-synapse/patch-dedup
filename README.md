# PatchDedup

<!-- V03 AUTO SUMMARY -->
Pipeline experimental em português para redundância entre realizações profissionais.
No teste reservado v03: **AP 0.965, F1 0.927**.
A base mDeBERTa NLI sem ajuste teve AP 0.984 e
F1 0.909. Dados sintéticos, sem revisão humana: resultados ainda
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
