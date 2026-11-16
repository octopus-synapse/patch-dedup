# Primeira bateria: ganho restrito ao piloto

CoSENT foi selecionado por AP de validação antes de qualquer teste. No teste
principal, com 72 pares de nove cenários, AP passou de 0,563 na base para 0,675;
F1 passou de 0,545 para 0,628. A precisão seletiva do selecionado foi somente 50%.
O refinamento com CosineSimilarityLoss obteve F1 0,733, mas não foi trocado pelo
vencedor depois de observar o teste, pois a seleção predefinida priorizava AP.

No desafio de 32 pares binários, AP foi 0,512 contra 0,495 da base; o efeito é
pequeno. Há também quatro casos ambíguos fora das métricas binárias. Os dados
foram gerados e revisados pela mesma IA, sem revisão humana.

`test.json`, `challenge.json`, `selection.json`, `selected-errors.json`, as
predições e os intervalos de bootstrap preservam os resultados reais. Estes
conjuntos se tornaram diagnósticos históricos ao iniciar a bateria v02.
