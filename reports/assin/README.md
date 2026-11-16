# ASSIN: avaliação externa e refinamento local

Comparação nos mesmos pares do teste pt-BR do ASSIN, com paráfrase como classe positiva.
O teste foi consultado em rodadas anteriores; os ajustes foram selecionados na validação.

| Etapa | AP | F1 | Precisão | Recall |
|---|---:|---:|---:|---:|
| Pipeline antes de treinar no ASSIN | 0.7660 | 0.6900 | 0.6423 | 0.7453 |
| Primeiro fine-tuning | 0.7875 | 0.7189 | 0.7027 | 0.7358 |
| Refinamento com exemplos difíceis e ranking | 0.8248 | 0.7429 | 0.7500 | 0.7358 |

O teste contém 2000 pares, dos quais 106 são positivos.
A configuração selecionada foi `binary-hard-ranking+incumbent:0.75`.
A validação ASSIN usada na última rodada tem 492 pares e 24 positivos; os 132 pares profissionais são desenvolvimento reutilizado.

## Evidências e reprodução

- [Métricas](metrics.json): três etapas no mesmo teste, com matrizes de confusão.
- [Seleção](selection.json): política, pesos, hashes e métricas de desenvolvimento.
- [Grafo](score-graph.json): caminhos relativos à raiz do projeto; não é um bundle publicado.
- [Validação](validation-candidates.json): candidatos e misturas avaliados antes do teste.
- [Treinos](training.json): configurações, épocas selecionadas e versões das bibliotecas.
- [Proveniência](protocol.json): revisão do dataset e hashes das fontes locais.

Para regenerar este snapshot a partir dos artefatos locais:

```bash
uv run --locked python scripts/export_assin_reports.py
```

Os pesos refinados e os textos ASSIN permanecem locais. A ficha do dataset no snapshot
registra licença desconhecida; este repositório não redistribui esse corpus nem atribui
a ele a licença dos dados sintéticos do projeto. A exposição da base durante pré-treino
não foi verificada. O modelo público continua sendo a v03 de configs/release.json.

## Interpretação

A queda em relação aos dados sintéticos mostra mudança de domínio e de distribuição.
Notícias não validam redundância entre realizações profissionais. O teste repetido é
diagnóstico e não substitui um novo teste independente com revisão humana.
O limiar foi escolhido na validação; F1 usa classificação binária em todos os pares.
Scores e a política de revisão não representam probabilidades calibradas.
Nenhuma técnica deve ser declarada benéfica isoladamente apenas por participar da mistura.
