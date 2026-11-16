# Relatórios de pesquisa

| Rodada | Evidência principal |
|---|---|
| [v01](v01/README.md) | Baselines, primeiros treinos, teste sintético e desafio |
| [v02](v02/README.md) | Bi-encoders, cross-encoders e holdout congelado |
| [v03](v03/README.md) | Transferência de NLI, seleção por grupos e pipeline publicado |
| [ASSIN](assin/README.md) | Avaliação externa, fine-tuning e refinamento local |
| [Currículos externos](real-resume-pilot/README.md) | Piloto em inglês com anotação por IA e diagnóstico local |
| [Docker](docker/README.md) | Verificação da imagem completa e inferência real offline |

O [registro de fundação](FOUNDATION.md) conserva o histórico inicial da engenharia
e do bloqueio de CI. Os JSONs de cada rodada preservam métricas, seleção, políticas,
hashes e proveniência. Resultados antigos permanecem como foram medidos.

As baterias profissionais v01/v02/v03 são sintéticas e revisadas pela IA autora, sem revisão
humana. O ASSIN usa notícias e outra proporção de positivos; suas métricas não são
comparáveis diretamente às dos pilotos sintéticos. O teste ASSIN já conhecido é
usado como diagnóstico repetido, com escolha de modelos e limiares na validação.

Pesos, datasets externos, caches e logs de execução ficam em `artifacts/`, fora do Git.
O snapshot ASSIN contém métricas e metadados portáveis; pode ser regenerado com
`uv run --locked python scripts/export_assin_reports.py` quando os artefatos locais
estiverem disponíveis. O refinamento local não substitui automaticamente o modelo publicado.
