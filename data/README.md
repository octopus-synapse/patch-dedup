# Dados sintéticos e proveniência

Todos os textos são fictícios, gerados e revisados pela mesma IA Codex. Não houve
revisão humana, concordância entre anotadores ou validação em currículos reais.
`review_status=reviewed` significa revisão pela IA autora, conforme os manifestos.

| Conjunto | Pares | Cenários | Uso |
|---|---:|---:|---|
| v01 treino | 336 | 42 | Ajuste inicial |
| Refinamento | 160 | 20 | Treino adicional; nunca validação/teste |
| v01 validação | 72 | 9 | Seleção de épocas, métodos e thresholds |
| v01 teste | 72 | 9 | Avaliação inicial; diagnóstico após v01 |
| Desafio | 36 | 20 | 32 binários e quatro ambíguos; diagnóstico após v01 |
| Holdout v02 | 48 | 24 | Novo teste congelado antes dos experimentos adicionais |

Treino refinado: 496 pares. Os dados têm 724 pares únicos no total (720 binários),
sem contar cópias agregadas em `all.jsonl`. As fixtures de software são separadas
e não entram em treino, seleção ou avaliação de ML.

`raw/*.psv` preserva os textos autorais de origem. `v01/`, `refinement/`,
`challenge/` e `v02/` contêm registros JSONL e manifestos SHA-256. A divisão v01
usa seed 42 e cenários completos; variantes do mesmo cenário ficam no mesmo split.
A construção compartilha padrões de redação entre grupos, uma limitação importante.

Positivo: mesma contribuição sem fato material exclusivo. Contribuições distintas,
sobreposição parcial e contradições são negativas; ambíguos têm alvo null e são
excluídos das métricas binárias. Números e contexto fazem parte da anotação.

Dados de pesquisa e fontes: CC BY 4.0. Atribuição: Enzo Ferracini Patti / PatchDedup,
com geração e revisão pela IA Codex. Veja `licenses/NOTICE.md` e o texto da licença.
Fixtures de software: MIT. Não há dados pessoais reais ou currículos de terceiros.
