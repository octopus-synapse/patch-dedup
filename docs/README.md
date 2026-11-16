# Documentação

O [README principal](../README.md) apresenta execução, resultados e limites atuais.
O [índice de relatórios](../reports/README.md) preserva as evidências de cada rodada;
a especificação inclui propostas que ainda não foram implementadas.

## Uso e desenvolvimento

| Documento | Conteúdo |
|---|---|
| [Contratos](CONTRACTS.md) | Entradas, saídas, scores e decisão |
| [Docker](DOCKER.md) | Imagem com checkpoint, lote, limites de recursos e verificação offline |
| [Avaliação](EVALUATION.md) | Splits, métricas e política de seleção |
| [Anotação](ANNOTATION_GUIDELINES.md) | Redundância, sobreposição, contradição e ambiguidade |
| [Dados](../data/README.md) | Proveniência e separação dos conjuntos |
| [Contribuição](../CONTRIBUTING.md) | TDD, ambiente e verificações |
| [Publicação](PUBLISHING.md) | Bundle, revisões e verificação no Hub |
| [Limites e ética](ETHICS.md) | Escopo do uso experimental |
| [Especificação](SPECIFICATION.md) | Requisitos originais e trabalho futuro |

## Pesquisa e reprodução

- [Métodos avaliados](TECHNIQUES.md).
- [Artigo v03](ARTICLE-v03.md) e [artigo inicial](ARTICLE.md).
- [Executor v03](AUTONOMOUS_RUN.md): etapas e retomada da publicação histórica.
- [Fine-tuning ASSIN](ASSIN-FINETUNING.md): primeira rodada com treino oficial.
- [Refinamento ASSIN](ASSIN-REFINEMENT.md): seis candidatos e comparação diagnóstica.
- [Scripts disponíveis](../scripts/README.md).

[README v02 arquivado](README-v02.md) registra a apresentação daquela versão e não
substitui os resultados atuais. O checkpoint público v03 e os refinamentos ASSIN
locais são identificados separadamente nos relatórios e na configuração de release.
