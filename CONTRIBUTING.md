# Desenvolvimento com TDD

Use Python 3.11 e `uv sync --locked`. O código e os identificadores usam inglês;
a documentação principal usa português. Não há dependência de ML no núcleo.

1. Escreva um teste de comportamento e execute-o; confira que a falha tem a causa esperada.
2. Implemente o mínimo necessário e execute os testes relevantes.
3. Refatore e rode a suíte, Ruff e verificação de formatação.
4. Faça um commit pequeno e verde, em inglês e com Conventional Commits.

Exemplos: `feat(data): detect scenario leakage`, `fix(cli): handle unreadable inputs`,
`test(contracts): cover invalid unicode`, `docs: explain annotation policy`.
Não publique commits vermelhos na branch principal. A falha RED é observada localmente;
o commit contém teste e implementação. Mudanças de documentação não exigem testes
artificiais. Conclua as verificações locais antes de enviar os commits e confira a CI quando
estiver disponível. O bloqueio de faturamento da conta pode impedir a execução
remota; registre esse estado separadamente dos resultados locais.

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build
```

Use `uv lock` ao alterar dependências e revise o diff do lockfile. O CI usa
`--locked` para rejeitar divergência de configuração. As dependências de ML são opcionais: extra `ml` para treino e `cpu` para CPU,
mutuamente exclusivos. Os testes de checkpoint real usam `PATCHDEDUP_MODEL` e
precisam de execução separada ou do workflow manual `model-smoke`.
Não confunda fixtures ou mocks com inferência real.

Mantenha pesos e credenciais fora do Git. Faça o split por cenário antes de formar
pares; registre autoria, geração, revisão e permissões. Não use o teste congelado
para escolher parâmetros. Métricas não serão critérios de testes unitários.

Os testes de software devem funcionar em um clone sem `artifacts/`. Use fixtures
temporárias para protocolos, caches e processos; os testes explicitamente marcados
`real_model` usam checkpoints locais. Não transforme metas de AP/F1 em asserções
de testes unitários. Para versionar resultados ASSIN, use
`scripts/export_assin_reports.py`: o exportador preserva métricas e hashes, sem
copiar dados externos, pesos ou arquivos de execução. Consulte o
[índice de documentação](docs/README.md) e o [guia de scripts](scripts/README.md).
