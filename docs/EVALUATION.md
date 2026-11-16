# Avaliação planejada e conclusão

Nenhum experimento de ML foi executado nesta etapa. Testes automatizados medem
comportamento do software; não demonstram capacidade de reconhecer redundância.

## Protocolo do próximo marco

1. Revisar piloto de 30–50 pares e taxonomia; expandir somente após verificar consistência.
2. Separar cenários, famílias e textos em treino/validação/teste; proposta 70/15/15 por grupos, seed 42. Registrar contagens e hashes reais.
3. Executar TF-IDF de caracteres 3–5, ajustado só no treino, e MiniLM original com revisão fixa.
4. Ajustar MiniLM com CosineSimilarityLoss, inicialmente até 3 épocas, batch 8 e learning rate 2e-5; selecionar por AP de validação.
5. Escolher thresholds de cada método na validação. Registrar F1 binário separadamente de cobertura e revisão; se nenhuma política segura funcionar, não habilitar decisões positivas.
6. Congelar seleção antes de avaliar no teste; publicar métricas e erros reais, inclusive resultados negativos.

Reportar precisão, recall, F1, macro-F1, AP, matriz de confusão, FPR em negativos
difíceis, cobertura e taxa de revisão. Reportar suporte por categoria e incerteza
por grupos quando viável. Não usar score como probabilidade. Checkpoints reais
terão smoke de forward/backward, save/reload e inferência; Docker será verificado
como usuário não root e sem rede. Não usar fixtures como avaliação científica.

## Checklist

- [x] Estrutura, pacote, lockfile e comandos de desenvolvimento.
- [x] Contratos, normalização e decisão com thresholds explícitos.
- [x] Auditoria de identidades, duplicatas e splits.
- [x] CLI offline de auditoria com JSON e exit codes.
- [x] Testes de contratos/dados/CLI e CI configurada.
- [x] Destinos públicos de GitHub, modelo e dataset criados.
- [ ] Piloto revisado e dataset com proveniência/licença.
- [ ] Baselines e fine-tuning com pesos efetivamente atualizados.
- [ ] Teste independente, thresholds selecionados e análise de erros.
- [ ] CLI de inferência e checkpoint publicado/recarregado por revisão.
- [ ] Docker CPU offline, cards finais e artigo com resultados.

A release estável `v0.1.0` depende dos MUST da especificação. API, Space e integração
com Patch Careers não fazem parte desta fundação. As publicações iniciais de cards
não contam como um segundo modelo treinado.
