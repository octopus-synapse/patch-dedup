# Direitos e atribuição

## Código e dados autorais

Código original: MIT, conforme [LICENSE](../LICENSE). Os dados sintéticos em
`data/raw/`, `data/v01/`, `data/refinement/`, `data/challenge/`, `data/v02/` e
`data/v03/` são CC BY 4.0, com autoria de Enzo Ferracini Patti e geração/revisão
pela mesma IA Codex. Não houve revisão humana independente. Atribuir ao projeto
PatchDedup e preservar a proveniência. Fixtures de software: MIT.

## Modelos-base

A linha NLI usa `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` e
`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`, cujas fichas registram MIT.
As revisões e modificações estão em [NLI-NOTICE.md](NLI-NOTICE.md), nos protocolos
e nos manifests dos bundles. Preservar a licença e os cards de origem.

O encoder de embeddings deriva de
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revisão
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, sob Apache-2.0 conforme a ficha de origem.
Atribuição: Sentence Transformers e autores do modelo-base. Preservar a licença,
os avisos e a descrição das modificações de fine-tuning.

## ASSIN local

O snapshot externo usa `nilc-nlp/assin`, revisão
`2d4aba156a914804e543cf2565aa133a107dce2a`. A ficha consultada registrava licença
desconhecida. Os textos externos e pesos derivados dessa rodada permanecem locais;
a licença dos dados sintéticos PatchDedup não se estende ao ASSIN.

[reports/assin](../reports/assin/README.md) versiona métricas, configurações e hashes,
sem redistribuir os textos do corpus. O modelo público v03 é identificado em
[configs/release.json](../configs/release.json), separado dos refinamentos ASSIN.

Não há afiliação ou endosso dos autores das bases, do dataset ou do iFood.
