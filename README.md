# PatchDedup

Detecção experimental de redundância entre realizações profissionais em português.
Compara pares de frases para identificar a mesma contribuição, informações distintas,
sobreposição parcial e contradições. O pipeline combina classificação por cross-encoder,
uma política de revisão e embeddings separados para similaridade por cosseno.

O projeto reúne preparação de dados, treinamento, seleção por validação, avaliação,
CLI e inferência local com Docker. Scores não são probabilidades calibradas.
A saída apoia revisão de texto; não exclui realizações nem classifica pessoas.

Para executar a imagem com o checkpoint embutido, veja o
[guia Docker](docs/DOCKER.md). Há também inferência JSONL em lote, com uma única
carga do modelo por execução.

## Executar

Requer Python 3.11 e [uv](https://docs.astral.sh/uv/). Para usar o checkpoint público,
instale as dependências de CPU e baixe a revisão fixa registrada em
[configs/release.json](configs/release.json). O download exige internet; a inferência é local.

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu hf download enzoferracini/patchdedup-ptbr-mdeberta \
  --revision f12d3936968d37b80a750c4a691e66b09edd83a9 --local-dir artifacts/model
uv run --locked --extra cpu patchdedup compare --model artifacts/model \
  --text-a 'Reduzi o tempo de resposta da API em 20%.' \
  --text-b 'Diminuí em 20% o tempo de resposta da API.'
```

A CLI executa o grafo de scores, a calibração e a política do bundle. Os pesos NLI
na raiz, carregados isoladamente pelo Transformers, retornam logits das classes NLI.
O cosseno vem do encoder em `embeddings/` e é informado separadamente do score de decisão.

Para comparar vários pares no mesmo processo:

```bash
uv run --locked --extra cpu patchdedup batch --model artifacts/model \
  --input examples/pairs.jsonl > resultados.jsonl
```

Cada linha de saída contém `line` e `result` ou `error`; textos inválidos não
interrompem os demais pares. A entrada também pode vir de stdin com `--input -`.
O contrato público declara português; suporte multilíngue das bases e controles
pontuais em inglês não constituem validação bilíngue.

## Resultados e limites

Os dados profissionais são sintéticos, gerados e revisados pela mesma IA, sem
revisão humana. No teste reservado v03 de 40 pares em 20 cenários:

| Método | Average Precision | F1 |
|---|---:|---:|
| Base mDeBERTa NLI sem ajuste | 0,9837 | 0,9091 |
| Pipeline v03 publicado | 0,9655 | 0,9268 |

O ajuste melhorou F1, mas reduziu AP frente à base forte. O teste pequeno e sintético
não demonstra generalização para currículos reais. [Relatório v03](reports/v03/README.md).

A avaliação externa usa os mesmos 2.000 pares de notícias do ASSIN pt-BR,
com 106 paráfrases positivas. Esses números pertencem a outro domínio e distribuição:

| Etapa local | Average Precision | F1 |
|---|---:|---:|
| Pipeline antes de treinar no ASSIN | 0,7660 | 0,6900 |
| Primeiro fine-tuning com treino ASSIN | 0,7875 | 0,7189 |
| Refinamento com exemplos difíceis e ranking | 0,8248 | 0,7429 |

A última mistura foi selecionada na validação. O teste ASSIN já havia sido consultado;
as medições repetidas são diagnósticas, não um novo teste independente. Notícias
também não validam redundância em currículos. [Resultados ASSIN](reports/assin/README.md).

**O refinamento ASSIN é local.** O checkpoint público baixado acima continua sendo
a v03. Os relatórios preservam os hashes e as configurações das duas linhas de pesquisa.

## Implementação e reprodução

- [Código](src/patchdedup): contratos, treinamento, scores, avaliação e CLI.
- [Testes](tests) e [contribuição](CONTRIBUTING.md): fluxo TDD e verificações locais.
- [Dados e proveniência](data/README.md): splits, anotação e fontes externas.
- [Configurações](configs/README.md): experimentos e revisão publicada.
- [Relatórios](reports/README.md): resultados v01/v02/v03 e ASSIN.
- [Scripts](scripts/README.md): reprodução, executores locais e manutenção.
- [Documentação](docs/README.md): contratos, métodos, limites e protocolos.

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build
bash scripts/docker_smoke.sh artifacts/model
```

Treinamento usa o extra `ml`; `cpu` e `ml` são mutuamente exclusivos. Dependências
estão fixadas em `pyproject.toml` e `uv.lock`. O núcleo de contratos e auditoria
funciona sem os extras de ML. Docker usa CPU, usuário não root e inferência offline.
A CI remota pode permanecer bloqueada pelo faturamento da conta; os checks locais
são registrados separadamente e nenhum orçamento foi aumentado.

[Modelo publicado](https://huggingface.co/enzoferracini/patchdedup-ptbr-mdeberta)
· [Dataset sintético publicado](https://huggingface.co/datasets/enzoferracini/patchdedup-synthetic-ptbr)

Código: MIT. Bases NLI: MIT; embeddings: Apache-2.0. Dados sintéticos: CC BY 4.0.
Dados ASSIN e pesos derivados permanecem locais; consulte [atribuição](licenses/NOTICE.md).
Projeto independente, sem afiliação ao iFood.
