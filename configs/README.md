# Configurações

`experiment.json` registra a primeira bateria. `experiment-v02.json` registra as
comparações adicionais e o novo holdout. Os valores não são recomendações universais;
foram definidos para estes pilotos locais. Resultados e políticas selecionadas
estão nos relatórios, com hashes dos conjuntos e revisões dos pesos.

Não copiar thresholds dos testes unitários. A inferência usa o `thresholds.json`
do bundle e valida sua integridade pelo manifesto. O extra `cpu` usa PyTorch CPU;
`ml` usa a instalação destinada ao treino. Escolha um deles em `uv sync --locked`.
