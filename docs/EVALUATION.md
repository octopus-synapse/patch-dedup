# Avaliação e refinamento

Os experimentos são locais e usam GPU NVIDIA RTX A1000 de 6 GB. Os dados foram
gerados e revisados pela mesma IA, sem anotação humana ou revisão independente.
Resultados descrevem estes conjuntos sintéticos; não demonstram qualidade em CVs reais.

## Primeira bateria — v01

480 pares de 60 cenários: 336 treino, 72 validação e 72 teste, com grupos
independentes. O refinamento acrescentou 160 pares de 20 cenários somente ao treino.
Cada cenário tem três positivos e cinco negativos difíceis. A auditoria confere
IDs e textos normalizados, mas não certifica independência de famílias sintáticas.

Comparações: TF-IDF de caracteres, MiniLM original, ajuste inicial com
CosineSimilarityLoss, continuação com essa loss e continuação com CoSENTLoss.
Três épocas por treino, seleção por AP de validação, depois F1, mantendo o primeiro
empate. Todos os parâmetros e pesos selecionados foram congelados antes do teste.
Um desafio separado tem 32 pares binários e quatro ambíguos.

Consulte `reports/v01/test.json`, `challenge.json`, `selection.json`,
`selected-errors.json` e `uncertainty.json`. O bootstrap reamostra cenários, não pares.
O teste principal tem apenas nove cenários: os intervalos são amplos. O método
selecionado por AP não precisa ser o melhor F1 depois de observar o teste.

## Bateria adicional — v02

Depois da primeira avaliação, seus testes se tornaram diagnósticos históricos.
Um novo holdout de 48 pares, 24 cenários, foi escrito e congelado antes dos novos
treinos. Sua concepção considerou dificuldades gerais já conhecidas; a autoria
continua sendo a mesma IA, portanto não há independência de anotador nem evidência
de generalização humana. O treino continua com 496 pares e a validação, com 72.

O protocolo versionado em `configs/experiment-v02.json` prevê:

- ContrastiveLoss com margem 0.5.
- OnlineContrastiveLoss com seleção de pares difíceis no batch e batch 16.
- CoSENT com amostragem balanceada e learning rate 5e-6.
- CoSENT congelando embeddings e as seis camadas inferiores, por quatro épocas.
- Cross-encoder com BCE ponderada, alternância AB/BA no treino e média bidirecional na inferência.
- Cross-encoder com embeddings e seis camadas inferiores congelados.

Os cross-encoders usam seis épocas; os demais novos treinos, três, salvo a exceção
explicitada. Os orçamentos não são iguais; não atribuir toda diferença somente à
loss. Também mudou a configuração de determinismo: v02 configura CuBLAS e desativa
attention kernels não determinísticos. v01 emitiu avisos de não determinismo CUDA.

O classificador adicional de características do par foi planejado, mas não
implementado nem avaliado após uma chamada de ferramenta bloqueada. Não aparece
como resultado. O desvio está registrado no relatório de desenvolvimento.

A seleção continua sendo AP de validação, seguida de F1 e primeiro empate. O
melhor bi-encoder e o melhor método de decisão são registrados separadamente.
Escolhas entre várias tentativas podem superadaptar a validação; a bateria inteira
e os resultados negativos devem ser reportados. O novo teste não será reutilizado
para escolher parâmetros ou trocar o vencedor.

## Métricas e decisão

Reportar precisão, recall, F1, macro-F1, AP, matriz de confusão, FPR em negativos
difíceis, cobertura, taxa de revisão e precisão seletiva. AP é Average Precision,
não área trapezoidal da curva PR. Threshold binário maximiza F1 na validação;
thresholds seletivos buscam precisão/NPV de 90% com pelo menos cinco exemplos por
direção. Isso não é calibração nem garantia de precisão no teste.

Mocks e transformers aleatórios minúsculos verificam o software, sem medir ML.
Smokes reais carregam o checkpoint treinado, conferem simetria e limites. O Docker
CPU deve executar sem rede, com modelo montado read-only e usuário não root.
