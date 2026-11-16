# Scripts de pesquisa e manutenção

Execute os scripts a partir da raiz do projeto, usando Python 3.11 e `uv`.
A CLI de inferência está no pacote `src/patchdedup/`; estes scripts coordenam
experimentos, verificações e preparação de artefatos.

## Fluxos principais

| Objetivo | Entrada |
|---|---|
| Verificar Docker CPU offline | `bash scripts/docker_smoke.sh artifacts/model` |
| Retomar a bateria v03 | `bash scripts/finish_remaining.sh --background` |
| Buscar metas conjuntas em desenvolvimento | `bash scripts/search_targets.sh --background` |
| Executar o primeiro fine-tuning ASSIN | `uv run --locked --extra ml python scripts/finetune_assin.py` |
| Retomar o refinamento ASSIN | `bash scripts/refine_assin.sh --background` |
| Exportar métricas ASSIN para o Git | `uv run --locked python scripts/export_assin_reports.py` |

Os executores retomam checkpoints concluídos e mantêm artefatos em `artifacts/`.
Um arquivo `comparison.json` existente representa uma rodada concluída; o mesmo
comando não cria outra rodada de treino. Consulte os protocolos em
[docs/ASSIN-FINETUNING.md](../docs/ASSIN-FINETUNING.md),
[docs/ASSIN-REFINEMENT.md](../docs/ASSIN-REFINEMENT.md) e
[docs/AUTONOMOUS_RUN.md](../docs/AUTONOMOUS_RUN.md).
Publicação no Hub é uma ação explícita do executor v03 com `--publish`, após checks.
Os executores ASSIN não publicam dados nem pesos.

## Etapas e histórico

- `study_v03.py`, `extra_v03.py`, `search_scores_v03.py`: treino e seleção em desenvolvimento.
- `calibrate_v03.py`, `evaluate_final_v03.py`: calibração reservada e avaliação congelada.
- `render_v03.py`, `export_v03.py`: relatórios/cards e bundle v03; o renderer preserva o README principal.
- `bootstrap_report.py`, `extend_experiment.py`, `evaluate_challenge.py`: bateria inicial e desafio.
- `compare_cross_encoders.py`, `select_v02.py`, `evaluate_v02.py`: comparações históricas v02.
- `prepare_release.py`: empacotamento histórico, anterior ao bundle v03.
- `parallel_assin_tail.py`, `sequential_assin_tail.py`: auxiliares de agendamento de uma
  execução local já ativa; não são necessários para iniciar ou retomar a fila sequencial.

Não executar dois controladores GPU em paralelo. Os auxiliares de agendamento
usam locks por candidato para evitar treinos duplicados e registram suas alterações.
Pesos, textos externos e logs não são exportados pelo gerador de relatórios ASSIN.
