---
language:
- pt
license: cc-by-4.0
task_categories:
- text-classification
size_categories:
- 1K<n<10K
configs:
- config_name: v03
  default: true
  data_files:
  - split: train
    path: v03/train.jsonl
  - split: validation
    path: v03/validation.jsonl
  - split: calibration
    path: v03/calibration.jsonl
  - split: test
    path: v03/test.jsonl
- config_name: refined
  data_files:
  - split: train
    path: refined/train.jsonl
  - split: validation
    path: refined/validation.jsonl
  - split: test
    path: refined/test.jsonl
- config_name: initial
  data_files:
  - split: train
    path: initial/train.jsonl
  - split: validation
    path: initial/validation.jsonl
  - split: test
    path: initial/test.jsonl
- config_name: challenge
  data_files:
  - split: test
    path: challenge/test.jsonl
- config_name: holdout_v02
  data_files:
  - split: test
    path: holdout_v02/test.jsonl
---
# PatchDedup Synthetic PT-BR

Pares profissionais **sintéticos, escritos e revisados pela mesma IA Codex, sem
revisão humana ou independente**. Dataset experimental, não representativo de
currículos reais. Não usar para avaliar candidatos ou comprovar qualidade de produção.

## Configurações e quantidades

| Configuração | Treino | Validação | Teste | Uso |
|---|---:|---:|---:|---|
| initial | 336 | 72 | 72 | Primeira bateria |
| refined, padrão | 496 | 72 | 72 | Treino com 160 pares adicionais |
| challenge | — | — | 36 | 32 binários e quatro ambíguos |
| holdout_v02 | — | — | 48 | Novo teste de 24 cenários para a bateria adicional |

Há 724 pares únicos, incluindo quatro ambíguos. Configurações initial/refined
compartilham exemplos deliberadamente; não somar seus tamanhos como dados únicos.
Os 480 pares iniciais pertencem a 60 cenários; 160 pares adicionais, a 20 cenários.
O desafio tem 20 grupos, e o holdout novo, 24. Manifestos SHA-256 estão em
`manifests/`; os textos de origem estão em `source/`.

```python
from datasets import load_dataset
train = load_dataset("enzoferracini/patchdedup-synthetic-ptbr", "refined", split="train")
```

## Anotação e campos

`redundant` → 1: mesma contribuição, sem fato material exclusivo.
`distinct`, `partial_overlap`, `contradictory` → 0. `ambiguous` → null: fora do
aprendizado e das métricas binárias. Sobreposição parcial é negativa para evitar
perder informação adicional. Contradição pressupõe o mesmo contexto/medida.

Campos: IDs de par/cenário/textos, text_a/text_b, relation, binary_label,
difficulty, domain, language, source_type, review_status, annotation_version e split.
`review_status=reviewed` significa revisão pela mesma IA autora. Não significa
anotação humana. Justificativa e protocolo estão no GitHub, sem alegar concordância
interanotador inexistente.

## Independência e limites

O split inicial usa seed 42 e cenários completos, antes de formar os pares.
Textos/IDs conhecidos não cruzam splits. Padrões de composição são compartilhados,
uma limitação que auditoria de hashes não resolve. O holdout v02 foi escrito e
congelado antes dos experimentos adicionais, após os testes v01 se tornarem
diagnósticos. Suas categorias gerais consideram limitações já observadas;
não é um teste produzido por autores independentes.

Não coletamos currículos, PII ou textos de candidatos. Os domínios incluem software,
operações, atendimento, comércio, educação, saúde, finanças, RH, marketing,
indústria, administração e serviços. A distribuição é artificial e os resultados
podem não generalizar. A prevalência positiva varia entre configurações.

## Direitos e referências

Dados e fontes: CC BY 4.0, com atribuição a Enzo Ferracini Patti / PatchDedup,
com geração e revisão pela IA Codex. Consulte LICENSE e provenance.json.
Projeto independente, sem afiliação ao iFood.

[GitHub, protocolos e relatórios](https://github.com/octopus-synapse/patch-dedup)
· [Modelo experimental](https://huggingface.co/enzoferracini/patchdedup-ptbr-minilm)

## Bateria v03

A configuração v03 inclui 744 pares de treino, 132 de desenvolvimento, 20 de
calibração e 40 de teste novo. Os 248 pares adicionais de treino são combinações
dos mesmos cenários; nenhum cenário de avaliação foi incorporado ao treino.
Pseudo-rótulos, quando presentes, estão em v03/pseudo_train.jsonl, não fazem parte
do treino padrão e não foram revisados por humanos. A coluna split dos registros
de calibração conserva validation por compatibilidade; o arquivo/manifesto
distingue seu papel reservado. Os dados de todas as versões são sintéticos.
