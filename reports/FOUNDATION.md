# Verificação da fundação — 15/11/2026 (America/Sao_Paulo)

A fundação `0.1.0.dev0` foi implementada e publicada. Há 68 testes passando
localmente; não há experimento de ML, checkpoint ou corpus revisado.

## TDD observado

| Incremento | RED observado antes da implementação | GREEN |
|---|---|---|
| Contratos e política | Import inicial ausente; depois 14 falhas de decisão/saída com implementação pendente | 44 testes passaram |
| Auditoria de dados | 17 testes falharam com auditoria pendente | 61 testes acumulados passaram |
| CLI como processo | 7 testes falharam com CLI pendente | 68 testes acumulados passaram |

Os commits funcionais incluem testes e implementação passando. O ciclo RED foi
executado localmente; não foram enviados commits intencionalmente quebrados.

## Verificação local

- `uv sync --locked`: ambiente resolvido a partir do lockfile.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked pytest -q`: 68 passaram.
- Ruff lint e checagem de formatação: aprovados.
- `uv build`: wheel e sdist `0.1.0.dev0` gerados.
- Wheel instalado em `/tmp/patchdedup-wheel-check`, fora do ambiente de desenvolvimento.
- CLI e import executados a partir de `/tmp`, usando o pacote em site-packages.
- Auditoria da fixture aprovada; PyTorch ausente no ambiente do wheel.
- `bash -n scripts/publish_cards.sh`: aprovado.

SHA-256 da fixture de software:
`e3417cabba8d1d2b06ccf261c41dba82777a72eea2d88bf7bd8846050601a7ae`.
Ela contém seis registros sem revisão e não serve como benchmark.

## Publicação conferida

Código funcional: commit
`d710e15bac4fb5c8fea845ecc8365af43a9b63fd`.
Primeiro push incluiu documentação e CI no commit
`8106cda54106b53b0ba5962dcf85bf8950b63476`.

| Destino | Revisão do card conferida |
|---|---|
| `enzoferracini/patchdedup-ptbr-minilm` | `214f0173f807d1e2fa21cbec57b75c88feb5a1b5` |
| `enzoferracini/patchdedup-synthetic-ptbr` | `0a621af136341c4c4f6ad36474d8a717bd18ac73` |

Os cards foram baixados por revisão fixa e comparados byte a byte com as fontes
locais. Cada destino contém somente README e `.gitattributes`, sem pesos ou dados.

## CI remota bloqueada pela conta

A [primeira execução](https://github.com/octopus-synapse/patch-dedup/actions/runs/43779395161)
foi marcada como falha antes de iniciar qualquer etapa. A anotação do GitHub diz:
“The job was not started because your account is locked due to a billing issue.”

Portanto, o workflow está configurado, mas a CI remota não foi validada em runner.
Após regularizar o bloqueio da conta, reexecutar `checks` e verificar o resultado.
Nenhuma configuração financeira foi alterada durante esta entrega.
