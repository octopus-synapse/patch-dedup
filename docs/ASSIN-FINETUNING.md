# Fine-tuning com ASSIN

A rodada `artifacts/assin-finetune/` combina exemplos profissionais do treino local
com o treino oficial ASSIN pt-BR. O executor treina dois candidatos, seleciona
checkpoints, mistura e limiar exclusivamente na validação e executa uma comparação
com o teste ASSIN pt-BR já avaliado anteriormente. O resultado não é uma nova
validação independente em currículos reais.

## Acompanhamento

```bash
cat artifacts/assin-finetune/status.json
tail -f artifacts/assin-finetune/runner.log
cat artifacts/assin-finetune/comparison.json
```

`comparison.json` é escrito ao concluir; contém `before`, `after`, configuração
selecionada e hash do protocolo. `predictions.json` conserva as previsões por par.
A unidade `patchdedup-assin-finetune.service` executa fora do terminal e da
conversa. O computador deve permanecer ligado e sem suspender.

Para retomar manualmente uma rodada interrompida, após verificar que não existe
outro executor ativo:

```bash
uv run --locked --extra ml --with pyarrow python scripts/finetune_assin.py
```

O controlador usa o mesmo lock do executor v03. Reaproveita candidatos completos;
renomeia diretórios de treino incompletos antes de reiniciar aquele candidato.
Não há retomada de minibatch/otimizador. O download precisa de rede; a inferência
usa checkpoints locais. `pyarrow` é uma dependência temporária do comando, sem
alterar as dependências de execução do pacote.

## Dados e seleção

A revisão do dataset vem de `artifacts/external-assin-ptbr/protocol.json` e é
fixada em `artifacts/assin-finetune/protocol.json`. O protocolo registra hashes
das fontes e dos JSONL preparados. Apenas PARAPHRASE é positivo. NONE e
ENTAILMENT são negativos para equivalência binária. A supervisão NLI direcional
é desativada para ASSIN, cuja anotação não fornece os dois alvos direcionais;
os exemplos profissionais mantêm sua supervisão original.

Frases compartilhadas após normalização de caixa e espaços são excluídas dos
conjuntos de prioridade inferior: teste preservado, depois validação, depois
treino. Isso não garante ausência de paráfrases próximas ou do mesmo evento
entre splits oficiais; também não verifica exposição no pré-treino da base.

A configuração inicial prepara 3.201 pares de treino, 624 de validação e 2.000
de teste, removendo 43 pares de treino e oito de validação por sobreposição.

Os candidatos começam no checkpoint `trial-0021` e na base mDeBERTa original.
Cada um recebe até três épocas, seed 137, seis camadas inferiores e embeddings
congelados, acumulação de gradientes, BF16, gradient checkpointing, warmup,
decaimento de aprendizado por camada, AdamW e parada antecipada. Os exemplos
profissionais ajudam a preservar a tarefa original. Não há promessa de melhora.

A mistura é escolhida numa grade fixa de cinco pesos pela maior AP de validação,
com F1 como desempate. O limiar é selecionado na validação. Os rótulos do teste
não entram no treinamento, na escolha de épocas, mistura ou limiar. O teste tem
106 positivos em 2.000 pares de notícias; não representa a distribuição de
currículos. Sua repetição mede a diferença nesta bateria conhecida.

Os dados e pesos ficam locais. A ficha ASSIN informa licença desconhecida; esta
rodada não publica dados, modelos ou previsões no Hub.
