# Configurações

| Arquivo | Escopo |
|---|---|
| [experiment.json](experiment.json) | Primeira bateria sintética, v01 |
| [experiment-v02.json](experiment-v02.json) | Comparações adicionais e holdout v02 |
| [experiment-v03.json](experiment-v03.json) | Treino NLI, desenvolvimento, calibração e teste v03 |
| [release.json](release.json) | Modelo/dataset publicados, revisões fixas e histórico |

Os parâmetros descrevem esses pilotos locais; não são recomendações universais.
A inferência carrega thresholds, grafo e manifesto do bundle. Não copiar thresholds
das fixtures de software nem escolher parâmetros a partir dos testes congelados.

Os protocolos ASSIN são preservados em [reports/assin/protocol.json](../reports/assin/protocol.json),
com configurações de treino em [training.json](../reports/assin/training.json) e
política/mistura em [selection.json](../reports/assin/selection.json). Esses checkpoints
são locais e não alteram as revisões de `release.json`.

Use Python 3.11 e `uv sync --locked --extra cpu` para CPU ou `--extra ml` para treino.
Os extras são mutuamente exclusivos e as dependências estão fixadas no lockfile.
