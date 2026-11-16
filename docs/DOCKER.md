# Inferência local com Docker

O container executa o checkpoint público v03 em CPU, sem rede durante a inferência,
como usuário não root. O modelo permanece experimental; empacotamento e testes de
software não demonstram generalização em currículos reais. Consulte o
[README](../README.md) para resultados e limites. O refinamento ASSIN é pesquisa local
e não está incluído nesta entrega pública.

Execute os comandos a partir da raiz do repositório. São necessários Docker com
BuildKit/buildx, Python 3.11, `uv` e espaço para a imagem e o checkpoint. O build
baixa dependências; somente o download inicial precisa acessar o Hugging Face.
As versões de Python, uv e dependências estão fixadas; o lockfile é obrigatório.

## Baixar o checkpoint público

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu hf download enzoferracini/patchdedup-ptbr-mdeberta \
  --revision f12d3936968d37b80a750c4a691e66b09edd83a9 --local-dir artifacts/model
```

A revisão é a registrada em `configs/release.json`. O carregamento verifica o
manifesto SHA-256. Pesos, caches e textos externos não entram no Git.

## Construir e conferir a imagem completa

```bash
PATCHDEDUP_BUNDLED=1 PATCHDEDUP_IMAGE=patchdedup:delivery-20261116 \
  PATCHDEDUP_SMOKE_OUTPUT=artifacts/docker-bundled-smoke \
  bash scripts/docker_smoke.sh artifacts/model
```

Esse comando prepara um contexto temporário só com os arquivos verificados pelo
manifesto e constrói o target `bundled`. Arquivos extras, caches e `.env` da pasta
de origem não são copiados. A imagem contém o checkpoint e as licenças; não precisa
de volume externo ou de acesso ao Hub em execução.

O smoke executa `batch` e `compare` reais com duas CPUs e limite de 4 GiB, verifica
simetria A–B/B–A, contrato JSON e tratamento de entrada inválida. Roda sem rede,
com filesystem somente leitura, `/tmp` temporário, capabilities removidas e usuário
65532:65532. As saídas e o ID da imagem ficam no diretório indicado. Os exemplos
são controles construídos; não medem a qualidade geral do modelo.

## Comparar um par

```bash
docker run --rm --network none --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges --memory 4g --cpus 2 \
  patchdedup:delivery-20261116 compare --model /model \
  --text-a 'Automatizei relatórios com Python.' \
  --text-b 'Usei Python para automatizar relatórios.'
```

A saída separa `similarity` (cosseno dos embeddings) de `decision_score` (score do
cross-encoder que orienta a política). Scores não são probabilidades calibradas.
A política pode retornar `redundant`, `not_redundant` ou `review`. O resultado apoia
revisão de texto; não exclui automaticamente realizações nem classifica pessoas.

## Comparar em lote

```bash
docker run --rm -i --network none --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges --memory 4g --cpus 2 \
  patchdedup:delivery-20261116 batch --model /model --input - \
  < examples/pairs.jsonl > resultados.jsonl
```

O modelo é carregado uma vez por processo. O arquivo de entrada contém um objeto
`CompareInput` por linha, com `text_a` e `text_b`. O idioma declarado é `pt-BR`;
`language=en` é rejeitado pelo contrato atual. Linhas em branco são ignoradas.

Cada linha de saída contém o número físico de linha e um `result` com o contrato
completo, ou um `error` seguro, sem repetir o texto enviado. O processamento
continua após erros de registro. Entradas acima dos limites de caracteres/tokens
são rejeitadas; não são truncadas. Há ainda limite de 64 KiB por linha JSONL.

Códigos de saída: 0 para todos os registros válidos; 1 se algum registro falhou;
2 para falha de inicialização/leitura ou entrada vazia. Uma saída parcial com exit
code 1 precisa ser inspecionada antes de incorporar os resultados. A execução em
lote reutiliza o modelo, mas processa os pares sequencialmente.

## Imagem com modelo montado

O target padrão continua sendo o runtime sem pesos:

```bash
bash scripts/docker_smoke.sh artifacts/model

docker run --rm --network none --read-only --tmpfs /tmp \
  --mount "type=bind,source=$PWD/artifacts/model,target=/model,readonly" \
  patchdedup:experimental compare --model /model \
  --text-a 'Automatizei relatórios com Python.' \
  --text-b 'Usei Python para automatizar relatórios.'
```

Para transportar a imagem completa sem registry, use `docker save` e `docker load`.
A imagem produzida pelo smoke é local; o script não publica imagem, modelo ou dados.
Para integração Python, os contratos e o comparador ficam em `src/patchdedup/`.
