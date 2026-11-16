# PatchDedup v03 — transferência de NLI e seleção por grupos

Resultado da política selecionada no teste reservado de 40 pares
em 20 cenários: **AP 0.965,
F1 0.927, macro-F1 0.925**.
A base mDeBERTa de NLI sem ajuste teve AP 0.984
e F1 0.909 no mesmo teste, com política fixada no desenvolvimento.
Não confundir o ganho sobre embeddings antigos com o ganho sobre essa base forte.

A seleção de scores avaliou 210 grafos em três folds
por cenário. A macro-F1 cross-fitted selecionada foi 0.946.
A seleção de épocas também usou desenvolvimento; cross-fitting não elimina todo
viés de seleção adaptativa. O teste só foi aberto após congelar modelo, grafo e calibração.

## Confiança e limitações

Cobertura fora de review: 0.725; precisão seletiva:
0.9230769230769231.
A calibração reservada tem somente 20 pares de dez grupos. A cobertura marginal
conformal observada no teste foi 0.700; sua
interpretação requer exchangeability e não garante precisão ou proteção a domain shift.
Os dados foram escritos e revisados pela mesma IA, sem revisão humana ou CVs reais.
Resultados são de um piloto sintético, não evidência de prontidão para produção.

## Candidatos e técnicas

| Candidato | AP desenvolvimento | F1 desenvolvimento |
|---|---:|---:|
| distilled_minilm | 0.861 | 0.840 |
| domain_adapted_minilm | 0.866 | 0.833 |
| hpo_1 | 0.967 | 0.931 |
| hpo_2 | 0.851 | 0.837 |
| hpo_4 | 0.976 | 0.947 |
| hpo_5 | 0.979 | 0.947 |
| hpo_5_quantized | 0.464 | 0.609 |
| hpo_6 | 0.933 | 0.891 |
| hpo_7 | 0.881 | 0.862 |
| lora_deberta | 0.925 | 0.891 |
| multitask_deberta | 0.945 | 0.914 |
| multitask_minilm | 0.860 | 0.831 |
| nli_deberta | 0.921 | 0.891 |
| nli_minilm | 0.804 | 0.803 |
| pseudo_distilled_minilm | 0.862 | 0.844 |
| seed137_minilm | 0.864 | 0.837 |

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


Relato técnico de portfólio, sem revisão acadêmica por pares.
