---
language:
- pt
tags:
- patchdedup
- experimental
- work-in-progress
---
# PatchDedup PT-BR MiniLM — em preparação

**Não há checkpoint neste repositório.** Esta página reserva o destino público do
experimento. Não é possível carregar um modelo PatchDedup com `SentenceTransformer`
a partir deste endereço neste momento. Nenhum fine-tuning ou benchmark foi concluído.

O objetivo é investigar redundância entre bullets profissionais em português:
paráfrases sem informação material exclusiva são positivas; contribuições distintas,
sobreposição parcial e contradições são negativas. Ambíguos serão excluídos do
aprendizado binário. Similaridade semântica não prova identidade de experiências.

A base proposta é
[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
sob Apache-2.0. Sua revisão será fixada antes dos experimentos. Ainda não há pesos
derivados, revisão de checkpoint, hardware, métricas ou thresholds operacionais.

A fundação contém contratos, normalização, decisão com thresholds explícitos e
auditoria de dados com TDD. Versão do software: `0.1.0.dev0`. Código sob MIT;
a licença do checkpoint derivado será confirmada antes de publicar os pesos,
preservando os avisos exigidos pela base.

[GitHub](https://github.com/octopus-synapse/patch-dedup)
· [Dataset em preparação](https://huggingface.co/datasets/enzoferracini/patchdedup-synthetic-ptbr)
· [Plano de avaliação](https://github.com/octopus-synapse/patch-dedup/blob/main/docs/EVALUATION.md)

Após o treino, esta página deverá registrar dados, proveniência, splits, revisão
da base, configurações, resultados contra baselines, erros, limites, uso reproduzível
e revisão dos pesos. O checkpoint produzirá embeddings; a decisão será uma política
externa de thresholds escolhidos na validação. O cosseno não será probabilidade.

Uso pretendido: sugestões para revisão textual humana, sem apagar conteúdo.
Não usar para avaliar candidatos, verificar fatos ou tomar decisões profissionais.
Projeto pessoal independente, sem afiliação ao iFood.
