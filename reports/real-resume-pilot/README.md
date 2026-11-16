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
Não há teste final independente nesta amostra. Não houve treinamento, inferência,
seleção de modelos ou acesso aos testes congelados do PatchDedup. Os dados não foram
integrados ao pipeline nem publicados. O piloto pode apoiar investigação de falsos
positivos; a ausência de positivos impede uma avaliação geral de redundância.
Não demonstra qualidade em português nem em inglês.

## Integridade dos artefatos locais

Verificados 100 IDs únicos, 20 pares por documento, 99 localizadores únicos de
trechos, correspondência entre CSV e JSONL e todos os hashes do manifesto.
Checks do projeto: `uv run --locked pytest -q` (147 passaram, um ignorado),
`uv run --locked ruff check .`, `uv run --locked ruff format --check .` e `uv build`.
O comportamento do pacote não foi alterado; os checks não validam os rótulos semânticos.

- `prepared/pairs.jsonl`: SHA-256 `899cbeeb3fe664452a875f791271ab5827a265ebf2329e02a4853615765deee1`.
- `prepared/review.csv`: SHA-256 `86d1379a597d6e6491081776b56dffb54c1c9c48bf0fde9421adf31e845515f4`.
- `annotations.psv`: SHA-256 `5e2044083c4aaa874d150d677f52797699fc9cbb967caa3100291aae8f054fff`.
