# Técnicas para melhorar modelos

Este catálogo é material de estudo, não uma lista de métodos executados no PatchDedup.
Qualidade preditiva, confiança, robustez e eficiência são objetivos diferentes.

| Frente | Técnicas e propósito |
|---|---|
| Qualidade dos dados | Correção de rótulos; adjudicação; deduplicação; revisão de ambiguidades; coleta direcionada; amostragem representativa |
| Aproveitamento de dados | Active learning; data augmentation; geração sintética revisada; weak supervision; pseudo-labeling; semi-supervised learning |
| Dificuldade e distribuição | Hard-negative mining; curriculum learning; balanceamento; class weights; oversampling; undersampling; SMOTE para dados adequados |
| Representação | Feature engineering; seleção de variáveis; normalização; imputação; PCA; embeddings; representações multimodais |
| Objetivo | Focal loss; contrastive loss; triplet loss; ranking losses; label smoothing; robust losses; cost-sensitive learning; multi-task learning |
| Otimização | Learning-rate tuning; warmup; schedulers; AdamW/SGD; ajuste de batch; acumulação; gradient clipping; learning rates por camada |
| Generalização | Weight decay; L1/L2; dropout; early stopping; congelamento; descongelamento gradual; EMA; SWA |
| Busca | Grid search; random search; otimização bayesiana; successive halving; Hyperband; population-based training; architecture search |
| Transferência | Transfer learning; continued pretraining; domain adaptation; full fine-tuning; LoRA; adapters; prompt/prefix tuning; distillation |
| Arquitetura | Árvores; boosting; redes convolucionais; Transformers; redes de grafos; bi-encoder; cross-encoder; mixture of experts |
| Combinação | Bagging; boosting; voting; averaging; stacking; blends; ensembles de seeds ou checkpoints |
| Confiança e decisão | Threshold tuning; Platt scaling; isotonic regression; temperature scaling; abstenção; detecção fora da distribuição; conformal prediction |
| Modelos generativos e sistemas | SFT; DPO; RLHF; RAG; reranking; ferramentas; decodificação restrita; verificação externa; test-time compute |
| Eficiência | Quantização; pruning; precisão mista; gradient checkpointing; compilação; caching; batching; speculative decoding |

Regularização combate memorização; melhor anotação combate rótulos incorretos.
Mais capacidade pode ajudar na subadequação e piorar o overfitting. Calibração muda
como interpretar a confiança, sem necessariamente aumentar F1 ou acurácia.
Quantização e caching buscam eficiência; não são garantias de melhores previsões.
RAG melhora acesso a informações externas, sem necessariamente alterar os pesos.

## Como verificar uma melhoria

Escolha a métrica e o protocolo antes de experimentar. Compare sob o mesmo split,
registre dados, seed, revisões e orçamento. Use validação para escolha e reserve o
teste para a avaliação final. Múltiplas seeds ajudam a medir estabilidade. Ablations
removem componentes para entender de onde veio um ganho. Se o teste passou a
orientar mudanças, ele se torna diagnóstico e é necessário outro holdout.

## Leituras

- [Ensembles — scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html)
- [Calibração — scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)
- [Losses — Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)
- [Cross-encoders — Sentence Transformers](https://sbert.net/docs/cross_encoder/training_overview.html)
- [LoRA — PEFT](https://huggingface.co/docs/peft/main/package_reference/lora)
- [Eficiência — Transformers](https://huggingface.co/docs/transformers/en/llm_tutorial_optimization)
