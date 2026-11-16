---
language:
- pt
tags:
- patchdedup
- experimental
- work-in-progress
---
# PatchDedup Synthetic PT-BR — em preparação

**Nenhum dataset de pesquisa está publicado neste repositório.** Há somente esta
Dataset Card inicial. Não há splits ou arquivos carregáveis com `load_dataset`.

O conjunto futuro reunirá pares de bullets profissionais em português brasileiro,
com cenários sintéticos ou autorizados e proveniência explícita. A quantidade,
distribuição, autoria, ferramentas de geração e revisão serão registradas após a
produção dos dados. Ainda não houve revisão humana de um corpus PatchDedup.

Taxonomia proposta: `redundant` → 1; `distinct`, `partial_overlap` e `contradictory`
→ 0; `ambiguous` → null, excluído do treino binário. Os splits serão por cenário,
com revisão de famílias de templates e identidades compartilhadas. Negativos
difíceis devem representar contribuições distintas sob contexto semelhante.

As seis fixtures do GitHub exercitam software e não são corpus de treino ou teste
científico. A licença do futuro dataset depende da auditoria de proveniência e
direitos de redistribuição; nenhuma licença de corpus é concedida por esta página.
Esta documentação original acompanha a licença MIT do projeto.

[GitHub](https://github.com/octopus-synapse/patch-dedup)
· [Modelo em preparação](https://huggingface.co/enzoferracini/patchdedup-ptbr-minilm)
· [Guia de anotação](https://github.com/octopus-synapse/patch-dedup/blob/main/docs/ANNOTATION_GUIDELINES.md)

Dados sintéticos podem compartilhar padrões artificiais e inflar resultados.
Documentar autoria/revisão e testar independência de cenários não comprova
representatividade de currículos reais. Não coletar ou publicar PII sem autorização.
Projeto pessoal independente, sem afiliação ao iFood; não destinado a avaliar pessoas.
