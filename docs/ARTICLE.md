# Beyond Semantic Similarity: Fine-Tuning Sentence Embeddings for Professional Achievement Redundancy in Brazilian Portuguese

## Resumo

Investigamos se especializar um encoder semântico permite reconhecer redundância
entre realizações profissionais em português sem eliminar informação material.
Construímos um corpus sintético, com cenários separados e negativos difíceis,
e executamos nove treinos locais de bi-encoders e cross-encoders. A primeira
bateria apresentou ganho de ranking no teste inicial, mas uma avaliação com novos
cenários e redação diferente não confirmou generalização confiável do bi-encoder
selecionado. O resultado ilustra a diferença entre melhorar validação e resolver
a tarefa. Este é um relato técnico de portfólio, sem revisão acadêmica por pares.

## Problema e definição operacional

Dois textos podem mencionar a mesma tecnologia sem comunicar a mesma contribuição.
Autenticação e autorização, por exemplo, são entregas diferentes. Um texto pode
incluir o fato do outro e acrescentar um método, escopo ou resultado relevante.
Nesse caso, considerar o par redundante poderia induzir perda de informação.

Definimos positivo como equivalência de contribuição sem informação material
exclusiva. Contribuições distintas, sobreposição parcial e contradições são
negativas. Casos ambíguos são excluídos do aprendizado binário. O sistema não
verifica fatos, avalia pessoas nem exclui conteúdo automaticamente.

## Dados e limites de independência

Os 480 pares iniciais derivam de 60 cenários escritos e revisados pela mesma IA.
A divisão por cenário, seed 42, produziu 336 pares de treino e 72 em cada conjunto
de validação e teste. Outros 160 pares de 20 cenários ampliaram somente o treino.
Um desafio separado reuniu 32 pares binários e quatro ambíguos.

O esquema preserva IDs, relações, alvo binário, domínio, dificuldade, origem e
status de revisão. A auditoria verifica duplicatas invertidas, identidades e
hashes entre splits. Isso não elimina padrões artificiais compartilhados na
construção. Não houve revisão humana nem concordância entre anotadores. Os dados
são deliberadamente difíceis e não representam a distribuição de currículos reais.

Após examinar a primeira avaliação, seus testes foram mantidos como diagnósticos.
Um novo holdout de 48 pares de 24 cenários foi congelado antes dos experimentos
adicionais. Sua elaboração considerou dificuldades gerais anteriores e manteve a
mesma autoria IA; portanto não equivale a um conjunto independente escrito por humanos.

## Métodos

A base é `paraphrase-multilingual-MiniLM-L12-v2`, com 384 dimensões e revisão fixa.
Comparamos TF-IDF de caracteres e o encoder original com fine-tuning supervisionado.
A primeira bateria treinou com CosineSimilarityLoss e depois comparou duas
continuações: essa mesma loss e CoSENTLoss. Os checkpoints foram selecionados
por AP na validação, com F1 como desempate e preferência pelo primeiro empate.

A bateria adicional comparou ContrastiveLoss, OnlineContrastiveLoss, amostragem
balanceada, learning rate menor e congelamento dos embeddings e das seis camadas
inferiores. Dois cross-encoders receberam os textos conjuntamente e usaram BCE
ponderada. A inferência do cross-encoder calcula as duas orientações e faz a média
dos scores sigmoid, preservando simetria. Esses scores não são cossenos ou
probabilidades calibradas. Orçamentos de épocas e batches variam, impedindo
atribuir todo ganho isoladamente à loss ou à arquitetura.

Todos os treinos ocorreram na GPU local RTX A1000 6 GB. Não houve aluguel de GPU,
endpoint, Space ou inferência paga. Configurações, dependências, hashes, revisões
e histórico por época acompanham o código. A bateria inicial emitiu avisos CUDA
de não determinismo; a adicional configurou CuBLAS e desativou kernels de atenção
não determinísticos. Isso não garante identidade numérica entre todos os hardwares.

## Resultados

Na bateria inicial, o CoSENT selecionado obteve AP 0,675 e F1 0,628 no teste,
comparados a 0,563 e 0,545 da base. No desafio separado, o ganho de AP foi pequeno:
0,512 contra 0,495. A precisão seletiva no teste principal foi de apenas 50%,
embora a validação sugerisse precisão maior. A hipótese de uma política segura
baseada apenas nesse conjunto de validação não foi sustentada.

Na segunda bateria, OnlineContrastiveLoss venceu na validação com AP 0,766,
contra 0,723 do incumbente. Porém, no novo holdout teve AP 0,397 e F1 0,516;
a base teve AP 0,369 e F1 0,531. O intervalo bootstrap por cenário para a diferença
de AP entre selecionado e base inclui zero: aproximadamente [-0,006; 0,078].
O score também ficou abaixo da prevalência positiva de 50%, sinalizando ranking
fraco nesse conjunto adversarial.

O cross-encoder com camadas congeladas apresentou AP 0,679 e F1 0,691 no novo
teste. Esse resultado sugere investigar a arquitetura conjunta com melhores dados.
Não trocamos retrospectivamente o vencedor predefinido: seria seleção pelo teste.
O candidato é preservado como alternativa de pesquisa, com essa distinção explícita.
A política seletiva desse cross-encoder não habilitou decisões positivas a partir
da validação, o que difere de suas métricas binárias. A tabela completa está em
`reports/v02/README.md` e as previsões brutas em JSON.

## Interpretação e próximos experimentos

Mais técnicas produziram conhecimento útil, mas não uma melhoria universal.
A mudança de redação e a composição do holdout expuseram uma validação pequena e
padrões sintéticos. Muitos exemplos são correlacionados dentro de poucos grupos;
o número de pares superestima o tamanho efetivo da evidência.

Os próximos investimentos devem priorizar anotação humana, critérios mais estáveis
de materialidade, estilos de escrita independentes e desenvolvimento por grupos
com maior diversidade. Novos métodos podem ser escolhidos em validação cruzada
por cenários, seguida de outro teste reservado. Não se deve continuar ajustando
hiperparâmetros até melhorar o teste já observado. Calibração e uso em produto
exigem dados representativos e avaliação adicional.

## Engenharia e reprodutibilidade

O desenvolvimento usou TDD para schemas, integridade, dados, decisão, CLI e treino.
Testes pequenos com Transformers aleatórios verificam software; smokes separados
carregam pesos realmente treinados. A CLI rejeita truncamento, verifica manifestos
e mantém scores e revisões explícitos. Docker CPU opera sem rede e como não root.
Os modelos e o dataset são publicados com suas licenças e limitações, sem inventar
métricas ou benefícios de negócio. Publicação não equivale a recomendação de uso.

## Referências

- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [Losses — Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)
- [Cross-encoder training](https://sbert.net/docs/cross_encoder/training_overview.html)
- [Modelo-base e licença](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [Configurações, dados e evidências](https://github.com/octopus-synapse/patch-dedup)
