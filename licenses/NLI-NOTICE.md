# Model sources and modifications

The primary NLI backbones are MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
(revision 8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c) and, for smaller candidate
components, MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli
(revision 0a71e92a985b6e1ad1828cf67ce9c459639c1dca). Their Hub cards declare MIT.
Credit the original model authors, Moritz Laurer and the referenced Microsoft/
XNLI/MNLI research authors. Preserve upstream-model-card.md and LICENSE.

Modifications, when selected: supervised directional NLI plus redundancy training,
LoRA merged into the full model, regularization, distillation, and/or the other
candidate-specific procedures in the attached evaluation and training reports.
A selected unmodified pretrained component must be identified as such in the
score graph; publication alone is not evidence of fine-tuning.

The optional embeddings/ encoder derives from sentence-transformers/
paraphrase-multilingual-MiniLM-L12-v2 and remains Apache-2.0, with its own LICENSE.
Dataset records authored for PatchDedup are CC BY 4.0. Code is MIT.
This is independent research, without affiliation or endorsement by iFood or
by any original model author. No private resumes were used.
