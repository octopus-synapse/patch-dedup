# Completar a bateria sem manter o agente ativo

No mesmo workspace, execute:

```bash
bash scripts/finish_remaining.sh --background --publish
```

O comando retorna um PID e os caminhos dos logs. Pode fechar a conversa e o
terminal. O computador precisa continuar ligado, conectado para downloads/uploads
e sem suspender. O processo é local: não cria jobs, endpoints, Spaces ou orçamento
pago. Os logins locais de `gh` e `hf` já configurados serão usados na publicação.

O executor espera os trabalhadores de pesquisa já iniciados, adquire uma trava
contra duas instâncias e pula checkpoints concluídos. Completa treino, LoRA,
distillation, busca TPE/Hyperband, adaptação de linguagem, pseudo-rótulos,
quantização, busca de ensembles/calibração, conformal, teste final, relatórios,
build/testes, Docker offline e publicação. Só então envia alterações geradas ao
GitHub com commit convencional em inglês. Não inclui pesos no Git.

```bash
# Inspecionar sem executar nem escrever estado
bash scripts/finish_remaining.sh --dry-run --publish
# Acompanhar (Ctrl+C aqui só para de acompanhar o log)
tail -f artifacts/v03/orchestration/runner.log
# Ler o estado atual
cat artifacts/v03/orchestration/status.json
```

O percentual do status é uma estimativa por marcos do controlador, não uma medição
exata do tempo total de pesquisa. Cada etapa tem seu log em
`artifacts/v03/orchestration/logs/`. O relatório final será `reports/v03/README.md`.

Se ocorrer falha, o status fica `failed` com o log correspondente. Execute o mesmo
comando para retomar. Diretórios interrompidos são preservados em
`artifacts/v03/aborted/`; checkpoints completos não são refeitos. Uma falha de
código ou dependência ainda pode exigir correção; o executor não usa um agente
para inventar reparos. Quantização sem suporte é registrada e os candidatos float
continuam disponíveis. Calibração/teste final possuem marcadores exclusivos;
não são reabertos para escolher parâmetros. Se a avaliação for interrompida depois
da abertura, a retomada exige inspeção para preservar a integridade experimental.

Sem `--publish`, a rotina prepara e verifica os artefatos localmente e não faz
upload ou push. Uma execução posterior com `--publish` usa os resultados existentes.
Não reinicie manualmente os scripts de treino em paralelo ao controlador.

A automação não substitui revisão humana dos dados. A fila de active learning
continua marcada como pendente. A CI remota pode permanecer bloqueada por
faturamento da conta GitHub; o executor não altera limites ou pagamentos.
