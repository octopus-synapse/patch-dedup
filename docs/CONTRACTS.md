# Contratos: entrada 0.1.0 e saídas 0.2.0/0.3.0

O pacote oferece `CompareInput`, `CompareOutput`, `PairRecord`, `Thresholds` e a
função `decide(score, thresholds)`. A inferência carrega somente bundles locais com manifestos verificados.
As definições executáveis estão em `src/patchdedup/schemas.py`.

## Entrada e política

`CompareInput` contém `schema_version="0.1.0"`, `language="pt-BR"`, `text_a` e
`text_b`. Textos são normalizados para Unicode NFC e espaços simples. Rejeita
texto vazio, Unicode não representável em UTF-8, mais de 1.500 caracteres após
normalização, idioma/versão não suportados e campos desconhecidos. Não altera
caixa, números, negações ou pontuação. O tokenizer valida o limite de tokens do
checkpoint, sem truncamento silencioso.

`Thresholds(low=..., high=...)` exige `-1 <= low < high <= 1`, sem defaults.
`decide` rejeita score não finito ou fora de `[-1,1]` com `INVALID_SCORE`.
Score `< low`: `not_redundant`; score `>= high`: `redundant`; restante: `review`.
Os limites vêm do bundle e do seu protocolo de seleção/calibração; não garantem
o mesmo desempenho em novos domínios.

`CompareOutput` segue os campos da seção 8 da especificação. Exige versões e
revisão não vazias, score válido, decisão coerente e `requires_review` equivalente
a `decision == "review"`. O produtor registra commit do Hub ou
hash do bundle em `model_revision`; o schema não consulta o Hub. Uma decisão
positiva continuará sendo sugestão para revisão humana.

## Registro de dados

Cada linha UTF-8 JSONL é um `PairRecord`. Veja a forma concreta nas fixtures;
elas não são exemplos anotados para treino.

| Campo | Contrato |
|---|---|
| `pair_id`, `scenario_id`, `text_a_id`, `text_b_id` | Strings não vazias; IDs de texto distintos dentro do par |
| `text_a`, `text_b`, `language` | Mesma validação textual da entrada |
| `relation` | `redundant`, `distinct`, `partial_overlap`, `contradictory`, `ambiguous` |
| `binary_label` | Inteiro estrito: 1 para redundant; null para ambiguous; 0 para os demais |
| `difficulty`, `domain`, `annotation_version` | Strings não vazias |
| `source_type` | `synthetic` ou `authorized` |
| `review_status` | `unreviewed`, `reviewed` ou `adjudicated` |
| `split` | `train`, `validation` ou `test` |

Um par representa um único cenário. Não crie pares entre cenários nesta versão;
divida cenários antes de construir pares. Para dados preexistentes que conectam
cenários, será necessária uma etapa adicional de componentes conexos.

## Auditoria e manifesto

`patchdedup validate-data --input arquivo.jsonl` espera um arquivo reunindo todos
os splits. Cada split deve conter exemplos binários positivos e negativos;
ambíguos entram nas contagens, mas não contam como classe binária. Linhas em
branco são ignoradas. O hash considera todos os bytes originais, inclusive espaços.

O manifesto tem `schema_version`, `valid`, `sha256`, `records`, `binary_records`,
`counts` por split/relação/revisão, `errors` e `warnings`. Cada erro indica código
e, quando aplicável, linha ou split. Contagens incluem apenas registros que
passaram no schema; erros impedem a aprovação do arquivo inteiro.

Códigos: `INVALID_UTF8`, `INVALID_JSON`, `INVALID_RECORD`, `EMPTY_DATASET`,
`MISSING_SPLIT`, `MISSING_CLASS`, `DUPLICATE_PAIR_ID`, `DUPLICATE_PAIR`,
`SCENARIO_LEAKAGE`, `TEXT_ID_LEAKAGE`, `TEXT_LEAKAGE`, `TEXT_ID_CONFLICT`.
Pares invertidos são a mesma identidade; mudanças de IDs não escondem pares de
textos repetidos. O hash textual usa a normalização NFC/espaços e preserva caixa.

O relatório não inclui conteúdo nem IDs do usuário. Avisos indicam registros sem
revisão e necessidade de revisar famílias de templates/quase duplicatas. O auditor
não certifica equivalência semântica, qualidade da anotação ou ausência de leakage
semântico. Ele mantém índices em memória; não é um processador de corpora ilimitados.

Erros de uso/leitura da CLI são JSON em stderr (`USAGE_ERROR`, `INPUT_UNAVAILABLE`),
com exit code 2. Relatório inválido sai em stdout com exit code 1; válido, com 0.

## Inferência implementada e score de decisão

`patchdedup compare --model <bundle> --text-a 'A' --text-b 'B'` roda em CPU por
padrão (`--device cuda` é opcional). Instale o extra `cpu` ou `ml`; eles são
mutuamente exclusivos. O primeiro usa wheels CPU, o segundo permite treino CUDA.

A saída 0.2.0 acrescenta `decision_source`, `decision_score` e
`decision_model_revision`. No bi-encoder, `decision_source=cosine` e os dois
campos de modelo separado são null. No cross-encoder, `similarity.score` continua
sendo cosseno dos embeddings; `decision_score` contém o score sigmoid não calibrado
que realmente orienta a política. A revisão do cross-encoder é registrada à parte.
Não apresentar esse score como cosseno ou probabilidade calibrada.

`Thresholds` também possui `positive_enabled` e `negative_enabled`, ambos true por
padrão. A seleção na validação pode desabilitar uma direção sem suporte suficiente;
nesse caso, os pares daquela região vão para review. A meta de precisão/NPV de 90%
com suporte mínimo de cinco pares é uma heurística experimental, sem garantia de
atingir essa precisão em novos dados. Os resultados v01 mostraram esse limite.

O carregamento verifica hashes dos arquivos do bundle. Os textos são tokenizados
sem truncamento: até 128 tokens por bullet no MiniLM atual; cross-encoder, quando
presente, também verifica o limite conjunto de 256 tokens. São feitas as duas
orientações do par e a agregação definida no bundle para preservar simetria.
O v03 publicado usa o mínimo dos scores NLI nas duas direções; cabeças binárias
usam a média dos sigmoids.
Nenhuma chamada a APIs externas é feita durante inferência.

## Inferência em lote

`patchdedup batch --model <bundle> --input pares.jsonl` aceita um `CompareInput`
por linha ou stdin com `--input -`. Reutiliza uma instância de `Comparator` e
processa os pares sequencialmente. Saída JSONL: `{"line": N, "result": ...}` ou
`{"line": N, "error": {"code": ..., "message": ...}}`. Não inclui os textos.
Linhas em branco são ignoradas, preservando a numeração física; linhas acima de
64 KiB recebem `INPUT_TOO_LONG` e são descartadas por inteiro, sem truncamento.

Exit codes: 0 se todos os registros foram processados; 1 se houve erros por registro;
2 para inicialização/leitura ou `EMPTY_INPUT`. Erros de registro ficam em stdout
junto aos resultados; erros globais ficam em stderr. Entradas inválidas não
interrompem os registros seguintes. Os limites do comparador e a declaração de
idioma `pt-BR` também se aplicam ao lote. Veja o [guia Docker](DOCKER.md).


## Pipeline v03

A saída `schema_version=0.3.0` usa `score_graph.json` para combinar os componentes
locais e aplicar a transformação de scores registrada. `decision_score.metric` é
`pair_score`; `similarity.metric` permanece `cosine`. `decision_model_revision`
identifica o hash do grafo e `model_revision` identifica o componente principal.
O manifesto verifica os arquivos dos componentes e do encoder de embeddings.

`prediction_set` registra os rótulos do conjunto conformal, enquanto `decision`
segue os thresholds do bundle. No v03 publicado, ambos derivam da calibração
reservada em 20 pares. Cobertura conformal pressupõe intercambiabilidade e não
garante precisão ou proteção contra mudança de domínio. A transformação identity
do grafo atual não produz probabilidades calibradas.
