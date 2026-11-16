# Publicação manual

GitHub mantém o código e as fontes dos cards em `hub/model/README.md` e
`hub/dataset/README.md`. O Hugging Face recebe somente os cards nesta etapa.
Os repositórios são independentes; não há espelhamento automático ou remoto Git
apontando o código inteiro para o Hub.

Destinos públicos aprovados:

- `octopus-synapse/patch-dedup` no GitHub.
- `enzoferracini/patchdedup-ptbr-minilm` como model repo no Hub.
- `enzoferracini/patchdedup-synthetic-ptbr` como dataset repo no Hub.

Use `gh api user --jq .login` e `hf auth whoami` para conferir as contas. A criação
do GitHub foi feita com `gh api --method POST orgs/octopus-synapse/repos`; a do Hub,
com `hf repos create <id> --type model|dataset --public`. Inspecione recursos
existentes antes de reutilizá-los e não sobrescreva conteúdo desconhecido.

Após revisar e commitar os cards, publique explicitamente:

```bash
bash scripts/publish_cards.sh
hf models info enzoferracini/patchdedup-ptbr-minilm
hf datasets info enzoferracini/patchdedup-synthetic-ptbr
```

O script usa a autenticação local e uma lista explícita de dois arquivos. Não
inclua tokens em argumentos, commits ou logs. Depois de publicar, confira conteúdo
e SHA de cada repositório; um upload de card não representa publicação de modelo.
Não há segredo do Hub na CI. Não foram criados Space, endpoint ou job pago.

No marco de pesos reais: salvar bundle completo, registrar commit de código e
hashes dos dados/configuração, incluir política validada e avisos da base, publicar
e testar recarregamento da revisão remota. No marco de dados reais: revisar
proveniência/licença, remover PII, congelar manifestos e atualizar a Dataset Card.
