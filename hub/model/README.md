---
language:
- pt
license: apache-2.0
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
datasets:
- enzoferracini/patchdedup-synthetic-ptbr
tags:
- patchdedup
- experimental
- pt-br
---
# PatchDedup — embeddings experimentais de redundância profissional

**Checkpoint real, com generalização insuficiente. Uso de pesquisa.** O modelo
selecionado por validação não manteve a melhoria no holdout novo: AP 0,397 e
F1 0,516. A precisão entre sugestões positivas foi apenas 0,333. Não usar para
apagar bullets, avaliar candidatos ou automatizar decisões sobre currículos.

## Modelo e uso

Bi-encoder de 384 dimensões, derivado do MiniLM multilíngue na revisão
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. A main contém o modelo ajustado com
OnlineContrastiveLoss, selecionado entre os bi-encoders pela AP de validação.
`SentenceTransformer.encode` retorna embeddings, não classes de redundância.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("enzoferracini/patchdedup-ptbr-minilm")
vectors = model.encode(["Automatizei relatórios com Python.",
                        "Usei Python para automatizar relatórios."],
                       normalize_embeddings=True)
score = float(vectors[0] @ vectors[1])
```

Para reprodução, use a revisão fixa registrada em `configs/release.json` do
[GitHub](https://github.com/octopus-synapse/patch-dedup). A CLI do projeto valida
o bundle e aplica `thresholds.json`; o score é cosseno, não probabilidade.
Nenhum texto é transmitido a APIs durante inferência local. Limite: 128 tokens
por bullet, sem truncamento silencioso. Não há suporte a inglês validado.

## Dados e treinamento

Dados sintéticos em português, escritos e revisados pela mesma IA Codex, sem
revisão humana. Treino refinado: 496 pares de 62 cenários; validação: 72 pares
de nove cenários. Cenários são separados antes dos pares; há construções
sintáticas compartilhadas, portanto a separação não garante independência de estilo.

Taxonomia: redundant positivo; distinct, partial_overlap e contradictory negativos;
ambiguous excluído do treino binário. O alvo é preservar informação material,
não apenas similaridade temática. Números, papéis e negação continuam difíceis.

Foram nove treinos locais em uma NVIDIA RTX A1000 6 GB. Seed 42, dependências
fixadas. A primeira bateria comparou CosineSimilarityLoss e CoSENT; a adicional
comparou ContrastiveLoss, OnlineContrastiveLoss, balanceamento, congelamento e
cross-encoders. Configurações, revisões, hashes e históricos acompanham os relatórios.

## Avaliação sem troca retrospectiva do vencedor

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

O holdout v02 tem 48 pares de 24 novos cenários e foi congelado antes da bateria
adicional. Os testes v01 já eram diagnósticos. O cross-encoder congelado foi melhor
no teste, mas não foi escolhido na validação; isso é resultado de avaliação,
não motivo para reescrever a seleção. A branch `cross-encoder-frozen` preserva
esse candidato alternativo quando publicada; `v01-incumbent` preserva a
inicialização dos experimentos v02.

AP mede ranking; F1 usa um threshold binário escolhido na validação. A política
seletiva usa thresholds diferentes, definidos na validação, e não alcançou a meta
de precisão no holdout. Não há calibração ou garantia de generalização. Consulte
`evaluation/v02/` e os [relatórios completos](https://github.com/octopus-synapse/patch-dedup/tree/main/reports).

## Limites, direitos e autoria

Somente dados sintéticos de um mesmo autor IA; validação pequena e muitas tentativas
podem superadaptar a seleção. Resultado não demonstra capacidade em currículos
reais. Não verifica fatos, não funde experiências e não avalia competência.

Pesos derivados: Apache-2.0, com LICENSE e NOTICE. Modificações: fine-tuning
supervisionado em pares e refinamentos descritos acima. Base: autores de
Sentence Transformers. Dados: CC BY 4.0. Código: MIT. Autoria: Enzo Ferracini
Patti com assistência do Codex. Projeto independente, sem afiliação ao iFood.
