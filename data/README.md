# Dados e proveniência

Os conjuntos profissionais versionados neste diretório são fictícios, gerados e
revisados pela mesma IA Codex. Não houve revisão humana, concordância entre
anotadores ou validação em currículos reais. `review_status=reviewed` significa
revisão pela IA autora, conforme os manifestos.

## Conjuntos sintéticos

| Conjunto | Pares | Cenários | Uso |
|---|---:|---:|---|
| v01 treino | 336 | 42 | Ajuste inicial |
| Refinamento | 160 | 20 | Treino adicional |
| v01 validação | 72 | 9 | Seleção inicial e desenvolvimento v03 |
| v01 teste | 72 | 9 | Diagnóstico histórico |
| Desafio | 36 | 20 | 32 binários e quatro ambíguos; diagnóstico |
| Holdout v02 | 48 | 24 | Teste congelado antes dos experimentos v02 |
| v03 desenvolvimento | 60 | 30 | Seleção, combinada aos 72 pares v01 de validação |
| v03 calibração | 20 | 10 | Calibração reservada após seleção |
| v03 teste | 40 | 20 | Teste reservado após congelar modelo e política |

O treino refinado original tem 496 pares. A função `augmented_training` adiciona
248 combinações dos mesmos 62 cenários de treino, totalizando 744 pares. Não são
novos cenários e nenhum cenário de avaliação é incorporado ao treino.

São 844 pares únicos autorais nas baterias v01/v02/v03, ou 1.092 incluindo as
248 combinações adicionais; cópias agregadas em `v01/all.jsonl` não são contadas
novamente. Fixtures de software ficam separadas e não entram nas métricas de ML.

`raw/*.psv` preserva as fontes autorais. Os diretórios dos conjuntos contêm JSONL
e manifestos SHA-256. O split v01 usa seed 42; a reserva v03 usa seed 3137.
Variantes do mesmo cenário ficam no mesmo split. Padrões de composição ainda são
compartilhados entre cenários, limitando a independência dos textos sintéticos.

Positivo: mesma contribuição sem fato material exclusivo. Contribuições distintas,
sobreposição parcial e contradições são negativas. Ambíguos têm alvo `null` e são
excluídos das métricas binárias. Números e contexto fazem parte da anotação.

## ASSIN externo

A pesquisa local também usa `nilc-nlp/assin`, revisão
`2d4aba156a914804e543cf2565aa133a107dce2a`, subconjunto pt-BR de notícias.
Paráfrase é positiva; implicação e nenhuma relação são negativas. Os rótulos ASSIN
não são convertidos em alvos NLI direcionais sem evidência da orientação.

Após excluir frases compartilhadas com os conjuntos reservados, o primeiro treino
usa 2.457 pares ASSIN e 744 sintéticos, totalizando 3.201. A validação mista tem
492 pares ASSIN e 132 sintéticos. O refinamento seguinte seleciona pela validação
ASSIN, relatando os 132 profissionais separadamente como desenvolvimento reutilizado.
O teste tem os mesmos 2.000 pares e 106 positivos das medições externas anteriores;
é diagnóstico conhecido e não um novo teste independente.

Os textos, caches e pesos dessa pesquisa ficam em `artifacts/`, fora do Git.
[reports/assin](../reports/assin/README.md) contém apenas métricas e metadados de
reprodução, incluindo revisão da fonte e hashes dos arquivos preparados. A ficha
da fonte no snapshot registra licença desconhecida; o corpus ASSIN não é
redistribuído nem recebe a licença dos nossos textos sintéticos.

## Piloto de currículos externos

Um [piloto local](../reports/real-resume-pilot/README.md) reúne 100 pares em inglês
de cinco currículos identificados como reais e anonimizados pelo publicador,
com anotação por IA e sem revisão humana. Textos, PDFs, contexto e manifestos ficam
em `artifacts/real_resume_pilot/`, fora do Git e do treinamento atual. A seleção
contém negativos e casos inconclusivos, sem positivos; não é um teste independente
nem valida desempenho bilíngue.

## Licenças

Dados sintéticos autorais: CC BY 4.0. Atribuição: Enzo Ferracini Patti / PatchDedup,
com geração e revisão pela IA Codex. Fixtures de software: MIT. Consulte
[atribuição e fontes](../licenses/NOTICE.md). Este diretório não contém currículos
de terceiros nem os textos externos do ASSIN.
