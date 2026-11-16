# Resultados v02: a melhoria na validação não se manteve

O método escolhido antes do teste foi **online_contrastive**. No holdout novo,
atingiu AP **0,397** e F1 **0,516**, frente a AP **0,369** e F1 **0,531** da base.
A diferença de AP não é conclusiva: bootstrap por cenário dá intervalo de 95%
para o ganho de aproximadamente **[-0,006; 0,078]**. A precisão seletiva foi apenas
**0,333**, com cobertura **0,375**. Este modelo não é recomendado para automação.

| Método | AP validação | AP holdout v02 | F1 holdout v02 |
|---|---:|---:|---:|
| lexical | 0.371 | 0.317 | 0.567 |
| base | 0.663 | 0.369 | 0.531 |
| incumbent | 0.723 | 0.375 | 0.657 |
| contrastive | 0.763 | 0.403 | 0.657 |
| online_contrastive | 0.766 | 0.397 | 0.516 |
| cosent_balanced | 0.753 | 0.374 | 0.585 |
| cosent_frozen | 0.761 | 0.384 | 0.657 |
| cross_encoder | 0.701 | 0.658 | 0.678 |
| cross_encoder_frozen | 0.677 | 0.678 | 0.691 |

O cross-encoder congelado teve o melhor resultado descritivo neste teste: AP
0,679 e F1 0,691. Não foi promovido retrospectivamente a vencedor da seleção:
a escolha original foi feita pela AP na validação. Sua política de validação
não habilitou decisões positivas; não confundir F1 binário com precisão seletiva.
Um checkpoint alternativo pode ser inspecionado como candidato experimental.

## Desenho e limitações

O holdout contém 48 pares de 24 cenários, metade positivos. A validação tem 72
pares de nove cenários, com prevalência positiva de 37,5%. Essa diferença de
composição e a mudança de redação tornam a avaliação mais exigente e expõem
fragilidade na seleção. Mesma IA gerou e revisou os dados, sem revisão humana.
Os novos cenários foram congelados antes destes treinos; dificuldades gerais da
primeira bateria influenciaram seu desenho. Não é validação independente humana.

Foram nove treinos: três na bateria v01, quatro novos bi-encoders e dois
cross-encoders. Foram comparados losses, learning rate, batch, balanceamento,
congelamento e arquitetura. Os orçamentos variam e não isolam causalmente todas
as mudanças. O classificador de features foi planejado e não executado.

## Evidências

- `selection.json`: escolha congelada antes de abrir o novo teste.
- `test.json`: métricas completas dos nove métodos comparados no holdout v02.
- `*-test-predictions.json`: scores e predições brutas, sem reajuste posterior.
- `selected-errors.json`: falsos positivos e falsos negativos do selecionado.
- `uncertainty.json`: bootstrap de 2.000 amostras de cenários, seed 42.
- `inference-example.json`: saída real da CLI com checkpoint selecionado.

Os relatórios v01 permanecem históricos. Seus testes não foram usados no treino
da bateria v02. Novos refinamentos exigem melhores dados de desenvolvimento e
outro teste reservado; não escolher hiperparâmetros pelo melhor resultado v02.
