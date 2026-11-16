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
artificiais. A integração contínua deve passar antes de concluir a entrega.

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build
```

Use `uv lock` ao alterar dependências e revise o diff do lockfile. O CI usa
`--locked` para rejeitar divergência de configuração. Futuras dependências de
ML devem ser opcionais; testes de checkpoint real precisam de execução separada.
Não confunda fixtures ou mocks com inferência real.

Mantenha pesos e credenciais fora do Git. Faça o split por cenário antes de formar
pares; registre autoria, geração, revisão e permissões. Não use o teste congelado
para escolher parâmetros. Métricas não serão critérios de testes unitários.
