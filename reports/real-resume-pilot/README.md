# Piloto local de currículos externos

Preparados 100 pares de trechos em inglês de cinco currículos identificados como
`real-anonymized` por Careerflow. A autenticidade não foi verificada independentemente.
A anotação é da IA Codex, sem revisão humana: 84 contribuições distintas, quatro
sobreposições parciais e 12 casos de contexto insuficiente. Não foram encontrados
positivos inequívocos ou contradições na seleção; nenhuma realização foi inventada
para completar classes. São 20 pares por documento, com reutilização de trechos.

A fonte é [Careerflow / ResumeBench](https://huggingface.co/datasets/Careerflow/ResumeBench),
revisão `529fa5cabee4723902daee2b5ba6b561c85c9533`, com licença CC BY 4.0 declarada
pelo publicador. Só a categoria `real-anonymized` foi utilizada. Cada trecho foi
conferido contra a extração do PDF, com normalização de espaços e variantes tipográficas.

Os artefatos ficam em `artifacts/real_resume_pilot/`, ignorados pelo Git:

- `README.md`: protocolo, atribuição, reprodução e limites.
- `prepared/review.csv`: revisão dos 100 pares com justificativas em português.
- `prepared/pairs.jsonl`: textos originais, contexto de emprego, localizadores e páginas.
- `manifest.json`: hashes, revisão e distribuição.
- `annotations.psv` e `build_pilot.py`: decisões e reconstrução determinística.

O alvo binário é **contextual**: empresa, cargo e período devem acompanhar os trechos.
Textos iguais de empregos diferentes não podem receber automaticamente o rótulo de
mesma contribuição. Casos de contexto insuficiente têm alvo nulo.

Todos os documentos foram atribuídos a desenvolvimento antes da construção dos pares.
Não há teste final independente nesta amostra. Na preparação não houve treinamento, inferência,
seleção de modelos ou acesso aos testes congelados do PatchDedup. Os dados não foram
integrados ao pipeline nem publicados. O piloto pode apoiar investigação de falsos
positivos; a ausência de positivos impede uma avaliação geral de redundância.
Não demonstra qualidade em português nem em inglês.

## Diagnóstico posterior do modelo local

O [resultado agregado](diagnostic.json) registra inferência real nos 100 pares usando
o refinamento ASSIN local, com grafo, pesos e limiares já selecionados. Esta execução
não avaliou o bundle v03 publicado. Nenhum texto foi truncado ou rejeitado.

| Grupo | Pares | Resultado |
|---|---:|---|
| Negativos anotados pela IA | 88 | 88 não redundantes; zero falsos positivos |
| Contexto insuficiente | 12 | 12 não redundantes; excluídos da contagem de acertos/erros |

O maior score foi 0,007877; o limiar binário permaneceu 0,494857. São scores de decisão
não calibrados, sem cálculo de cosseno. Sem positivos e com cinco grupos dependentes,
esse resultado não mede sensibilidade nem comprova qualidade geral em inglês.

Seis controles construídos à parte verificaram a execução: duas duplicações exatas
em inglês, duas paráfrases simples (inglês e português) e dois negativos (um por
idioma). Os quatro positivos foram classificados como redundantes e os dois
negativos como não redundantes. São checagens pontuais, não um benchmark bilíngue;
não foram misturados aos 100 pares reais.

A revisão pela mesma IA autora manteve os rótulos provisórios e os 12 inconclusivos.
Os arquivos locais `diagnostics/local-assin-v1/analysis.md`, `review.csv`,
`predictions.json`, `protocol.json` e `manifest.json` preservam análise, previsões e
proveniência. O CSV foi reexportado em UTF-8 após uma falha de codificação, usando
as previsões salvas, sem repetir a inferência. Os hashes dos insumos foram
reverificados. Não houve treino, alteração de limiares ou leitura de testes congelados.

Reprodução: `uv run --locked --extra ml python scripts/diagnose_resume_pilot.py
--device cuda --output artifacts/real_resume_pilot/diagnostics/nova-rodada`.
O executor exige diretório novo, preserva contexto e separa inconclusivos e rejeições.
Testes de software cobrem essas separações e exportação Unicode em locale ASCII;
essas fixtures não medem qualidade do modelo.
Após o diagnóstico: 151 testes passaram, um foi ignorado; Ruff e build passaram.
Também foram conferidos os 100 scores contra a soma ponderada dos componentes,
os limites de tokens, a planilha e todos os hashes dos insumos e artefatos.

## Integridade dos artefatos locais

Verificados 100 IDs únicos, 20 pares por documento, 99 localizadores únicos de
trechos, correspondência entre CSV e JSONL e todos os hashes do manifesto.
Checks do projeto: `uv run --locked pytest -q` (147 passaram, um ignorado),
`uv run --locked ruff check .`, `uv run --locked ruff format --check .` e `uv build`.
O comportamento do pacote não foi alterado; os checks não validam os rótulos semânticos.

- `prepared/pairs.jsonl`: SHA-256 `899cbeeb3fe664452a875f791271ab5827a265ebf2329e02a4853615765deee1`.
- `prepared/review.csv`: SHA-256 `86d1379a597d6e6491081776b56dffb54c1c9c48bf0fde9421adf31e845515f4`.
- `annotations.psv`: SHA-256 `5e2044083c4aaa874d150d677f52797699fc9cbb967caa3100291aae8f054fff`.
