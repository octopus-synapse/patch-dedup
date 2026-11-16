# Refinamento dos classificadores de equivalência

Esta rodada compara seis candidatos locais com o checkpoint `continued` da
primeira rodada ASSIN. A execução não depende da conversa. Os modelos publicados
permanecem como referência até uma decisão explícita de publicação.

```bash
bash scripts/refine_assin.sh --background
cat artifacts/assin-refinement/status.json
tail -f artifacts/assin-refinement/runner.log
```

O serviço de usuário é `patchdedup-assin-refinement.service`. O computador precisa
ficar ligado e sem suspender. Após reiniciar, o mesmo comando retoma checkpoints
concluídos; uma época interrompida pode precisar ser refeita. Arquivos parciais são
preservados em diretórios `interrupted-*`, sem sobrescrever pesos. Um lock
compartilhado impede dois controladores locais de usarem a GPU ao mesmo tempo.
Para parar: `systemctl --user stop patchdedup-assin-refinement.service`.

## Comparações previstas

São no máximo seis treinos de quatro épocas, com parada antecipada e limite de
12 horas para o treinamento. A seleção e avaliação ainda podem levar alguns
minutos adicionais. É possível configurar `--epochs` antes da primeira execução;
protocolos, dados, configurações e código ficam fixados por SHA. Alterações no
protocolo exigem outro diretório `--output`. Aumentar `--max-hours` permite ampliar
o orçamento antes de congelar a seleção, sem alterar os candidatos.

| Candidato | Objetivo | Amostragem |
|---|---|---|
| binary-natural | BCE, nova cabeça binária | Natural, peso de positivos |
| binary-balanced | BCE, nova cabeça binária | Metade positivos por lote |
| binary-ranking | BCE + ranking logístico | Metade positivos por lote |
| binary-hard-ranking | BCE + ranking logístico | Balanceada e ponderada por dificuldade |
| assin-three-way | CE: nenhuma relação, implicação, paráfrase | Metade paráfrases por lote |
| binary-seed42 | BCE + ranking, outra seed, menos camadas congeladas | Balanceada e ponderada |

As duas ordens de cada par recebem supervisão simétrica. A cabeça de classificação
é reinicializada, inclusive no caso de três saídas: os significados das classes
ASSIN não são os mesmos das antigas classes NLI direcionais. A inferência usa a
média das probabilidades da classe positiva nas duas ordens. Scores continuam
sem calibração probabilística.

A amostragem balanceada garante positivos em cada lote físico de quatro pares,
sem combinar essa compensação com o antigo peso de positivos da BCE. O aumento
de dificuldade usa exclusivamente erros do modelo anterior nos **pares de treino**;
não adiciona exemplos do teste nem pseudo-rótulos. Os candidatos usam AdamW,
warmup/cosseno, taxas por camada, congelamento, BF16, checkpointing e consistência
entre ordens. São ablações de combinações; não demonstram isoladamente a contribuição
de cada regularizador compartilhado.

## Dados e avaliação

Os 3.201 pares de treino são reutilizados, com revisão dos hashes e da separação
entre treino e validação. A escolha usa a parte ASSIN da validação preparada na
rodada anterior; os 132 pares profissionais ficam como diagnóstico de possível
regressão de domínio, relatado separadamente. O protocolo registra os tamanhos e
quantidades de positivos efetivamente encontrados. Os exemplos continuam com a
proveniência original; não houve anotação humana nova.

O checkpoint anterior participa como candidato. Modelos isolados e uma grade pequena
de misturas são comparados pela validação. Uma alternativa só é escolhida se AP e
F1 de validação não regredirem em relação ao anterior; entre elegíveis, vence AP,
depois F1. Isso não garante que precisão, recall ou métricas de teste melhorem.
Pesos e limiar ficam congelados antes da avaliação nos mesmos 2.000 pares ASSIN.

Esse teste já foi consultado e motivou a estratégia da pesquisa. É uma comparação
diagnóstica repetida, não um novo teste independente. Não é usado para escolher
épocas, candidatos, misturas ou limiares. Se uma execução falhar após a seleção,
a retomada usa a seleção congelada, sem refazer a busca a partir do teste.

Ao terminar:

```bash
cat artifacts/assin-refinement/comparison.json
```

`selection.json`, `score-graph.json`, `protocol.json`, `validation-candidates.json`,
`predictions.json` e `README.md` preservam as escolhas, os resultados e as limitações.
Nenhum serviço pago, download adicional ou publicação automática é iniciado.

## Correção histórica da loss de ranking

Uma regressão executava duas vezes a contribuição de ranking no treinador NLI.
O teste passou a verificar uma contribuição por lote, e a duplicação foi removida.
A busca histórica `target-search` com peso de ranking não nulo teve contribuição
efetiva duplicada; seus scores e checkpoints são preservados, sem reescrever os
resultados. A primeira rodada ASSIN tinha `ranking_weight=0`, portanto não foi
afetada. A nova rodada tem testes explícitos da magnitude da contribuição.
