---
language:
- pt
license: apache-2.0
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
datasets:
- enzoferracini/patchdedup-synthetic-ptbr
---
# PatchDedup — candidato cross-encoder congelado

Alternativa experimental avaliada, **não selecionada pela validação**. A main
preserva o bi-encoder escolhido antes do teste. Esta branch disponibiliza o
cross-encoder com embeddings e seis camadas inferiores congelados, sem promover
retrospectivamente um vencedor por olhar o teste.

No holdout v02 de 48 pares de 24 cenários: AP 0,679, F1 binário 0,691, precisão
0,613, recall 0,792. Na validação: AP 0,677. O threshold binário foi escolhido
na validação. A política seletiva desabilitou decisões positivas por falta de
precisão/suporte suficiente na validação: a CLI retorna review ou not_redundant,
e não o rótulo binário usado para calcular F1. Não usar para automação.

O bundle contém embeddings MiniLM OnlineContrastive na raiz e o modelo conjunto
em reranker/. SentenceTransformer.encode retorna somente embeddings; não executa
o reranker. A CLI patchdedup compare aplica ambos: similarity.score é cosseno;
decision_score é a média de scores sigmoid nas orientações AB e BA, não calibrada.
As revisões dos dois modelos são registradas separadamente.

Treino local em RTX A1000 6 GB, 496 pares sintéticos, seis épocas, BCE ponderada,
learning rate 3e-5, seed 42. Base MiniLM na revisão
e8f8c211226b894fcb81acc59f3b34ba3efd5f42, pesos derivados Apache-2.0.
Os dados foram escritos e revisados pela mesma IA, sem revisão humana. A validação
é pequena e há padrões artificiais. Resultados não representam currículos reais.

Baixe esta revisão para uma pasta local e use a CLI do projeto. O Docker CPU
executa offline, como não root. Limites de tokens são verificados sem truncamento.
Consulte configs/release.json para o SHA fixo e os relatórios completos:
https://github.com/octopus-synapse/patch-dedup/tree/main/reports/v02

Projeto independente, sem afiliação ao iFood. Não verifica fatos ou avalia pessoas.
Não apaga experiências. LICENSE e NOTICE preservam direitos da base e modificações.
