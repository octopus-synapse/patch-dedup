# Publicação, versões e reprodução

O GitHub mantém fontes, dados e relatórios. O Hub recebe bundles explicitamente
preparados, sem espelhar a árvore de trabalho ou incluir credenciais.

## Destinos e variantes

- Código: `octopus-synapse/patch-dedup`.
- Modelo: `enzoferracini/patchdedup-ptbr-minilm`, main com a seleção v02.
- Branch `v01-incumbent`: inicialização fixa dos experimentos adicionais.
- Branch `cross-encoder-frozen`: candidato alternativo, não escolhido pela validação.
- Dataset: `enzoferracini/patchdedup-synthetic-ptbr`, com configurações initial, refined, challenge e holdout_v02.

As revisões exatas ficam em `configs/release.json`. O modelo principal foi
selecionado antes do teste; preservar uma alternativa não muda essa seleção.
Não interpretar a disponibilidade no Hub como recomendação de produção.

## Preparar e publicar manualmente

```bash
uv run --locked --extra cpu python scripts/prepare_release.py
hf upload enzoferracini/patchdedup-ptbr-minilm artifacts/publication/model . \
  --commit-message 'feat(model): publish experimental refined sentence encoder'
hf upload enzoferracini/patchdedup-synthetic-ptbr artifacts/publication/dataset . \
  --type dataset --commit-message 'feat(data): publish synthetic grouped benchmark and provenance'
```

O staging usa uma lista explícita de arquivos e verifica hashes do bundle. Os
cards fazem parte do manifesto: não atualizar somente o README remoto depois de
publicar; reconstruir e conferir o manifesto junto. A alternativa de cross-encoder
inclui o mesmo bi-encoder e o subdiretório `reranker/`, com score de decisão
separado. Não carregar essa pasta como se o cosseno fosse a decisão do reranker.

Verificar após upload: baixar por commit fixo, comparar arquivos, verificar
manifesto e executar inferência/testes offline. O release deve registrar os SHAs
reais. Publicar somente dados sintéticos/autorizados sob a licença documentada.
Não incluir tokens em comandos ou commits. Não há publicação automática na CI.

## Reproduzir a bateria adicional

Instale `uv sync --locked --extra ml`. Baixe o MiniLM base na revisão fixa de
`configs/experiment-v02.json` e o checkpoint histórico na revisão
`incumbent_revision` de `configs/release.json`. O segundo vai para
`artifacts/run-42/selected-model`, a inicialização usada na bateria adicional.

```bash
uv run --locked --extra ml python scripts/extend_experiment.py \
  --output artifacts/new-run --reports artifacts/new-run-reports --device cuda
uv run --locked --extra ml python scripts/compare_cross_encoders.py \
  --output artifacts/new-run --reports artifacts/new-run-reports --device cuda
uv run --locked --extra ml python scripts/select_v02.py \
  --output artifacts/new-run --reports artifacts/new-run-reports --device cuda
# Só depois da seleção congelada; o teste não serve para escolher o vencedor.
uv run --locked --extra ml python scripts/evaluate_v02.py \
  --output artifacts/new-run --reports artifacts/new-run-reports --device cuda
```

Use diretórios novos; os scripts rejeitam sobrescrita. O teste possui marcador
exclusivo antes da leitura e não deve ser reaberto para selecionar parâmetros.
Comparar resultados a partir de predições já salvas ou reamostrá-las por cenário
não corresponde a outro treino. A inicialização histórica precisa ser a revisão
publicada, não um novo treino com versão de código diferente.

Para a bateria v01, `python -m patchdedup.experiment develop` e `final` usam a
configuração original; consultar os commits e hashes de fonte dos relatórios para
reproduzir o comportamento histórico. Versões antigas de kernels CUDA podem
produzir diferenças; não prometemos reprodutibilidade bit a bit em outro hardware.

## Custos e CI

Foram usados treino local, Docker local e repositórios públicos. Nenhum endpoint,
Space, job de GPU, serviço pago ou orçamento adicional foi habilitado. A conta
GitHub bloqueou Actions por faturamento antes de iniciar os jobs. Não alteramos
pagamentos ou limites. A validação local e as revisões publicadas são registradas;
a CI remota fica pendente até a conta permitir execução.
