# Guia de anotação 0.1.0 — proposta para o piloto

Um positivo comunica a mesma contribuição sem informação material exclusiva.
Tema, tecnologia ou projeto compartilhados não bastam para redundância.

| Relação | Critério | Alvo binário |
|---|---|---|
| redundant | Paráfrase da mesma contribuição sem perda de fatos | 1 |
| distinct | Entregas, ações ou objetos diferentes | 0 |
| partial_overlap | Há núcleo comum e informação material exclusiva | 0 |
| contradictory | Alegações incompatíveis sob o mesmo contexto conhecido | 0 |
| ambiguous | Contexto insuficiente ou desacordo não resolvido | null; excluir do treino binário |

“Implementei OAuth” e “Implementei OAuth e recuperação de senha” são sobreposição
parcial. “Reduzi a latência em 30%” e “Reduzi a latência em 50%” são contraditórios
somente quando representam a mesma medida/evento. “Liderei” e “participei” não
são sinônimos de papel. Preserve métodos, escopo, resultados, números e negações.

Crie cenários fictícios com fatos de origem; aloque-os em splits antes de formar
pares. Registre justificativa, fonte, ferramenta/prompt de geração, versão e
responsável pela revisão no manifesto de anotação. Não envie justificativas como
features do modelo. Varie domínio, sintaxe e dificuldade sem usar palavras que
revelem o rótulo. Famílias de templates também precisam de separação/revisão.

Comece com 30–50 pares, incluindo negativos difíceis e paráfrases de baixa
sobreposição lexical. Revise sem ver a sugestão de rótulo quando possível;
adjudique desacordos. Se houver apenas um revisor, faça segunda revisão cega e
registre essa limitação. Não declare revisão humana quando ela não ocorreu.
O piloto só vira benchmark após revisão, auditoria e congelamento dos manifestos.
