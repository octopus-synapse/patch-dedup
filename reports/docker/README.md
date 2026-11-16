# Verificação local da entrega Docker

Em 16/11/2026 foram construídas e executadas duas variantes com o checkpoint público
v03: runtime com modelo montado e imagem completa com pesos embutidos.
[Metadados e hashes](verification.json) registram IDs das imagens, revisão do modelo,
manifesto e hashes do código usado. Não houve novo treinamento nem mudança de limiares.

A imagem completa local é `patchdedup:delivery-20261116`; foi executada sem volume
externo de modelo. Os dois smokes passaram com `--network none`, filesystem somente
leitura, usuário 65532:65532, duas CPUs e limite de 4 GiB. PyTorch é CPU-only e os
arquivos em `/model` correspondem exatamente ao manifesto mais o próprio manifesto.
A imagem não foi enviada a um registry.

As verificações executaram `batch` e `compare` reais, conferindo JSON, simetria,
separação entre cosseno e score de decisão e recuperação após registro inválido.
A paráfrase demonstrativa recebeu `redundant` nas duas orientações; o par de
contribuições distintas recebeu `not_redundant`. São exemplos construídos para
smoke, não novas métricas ou evidência de generalização.

Foram aprovados 160 testes de software (um ignorado por exigir configuração de
checkpoint), mais o teste explícito de checkpoint real, Ruff, formatação, sintaxe
do shell e `uv build`. Os testes unitários usam fixtures de software; a inferência
real foi verificada separadamente com os pesos publicados. O workflow manual
`model-smoke` foi atualizado para incluir o Docker; não foi executado remotamente
nesta entrega e não se afirma aprovação da CI remota.

Os logs completos ficam fora do Git em `artifacts/docker-smoke/` e
`artifacts/docker-bundled-smoke/`. Consulte o [guia de execução](../../docs/DOCKER.md).
O modelo permanece experimental, com dados profissionais sintéticos e avaliação
limitada. Empacotamento operacional não comprova confiabilidade em produção.
