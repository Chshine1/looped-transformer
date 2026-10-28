# Looped Transformer for Natural-Language Inference

This repository studies whether a recurrent-depth Transformer can learn strong,
sample-dependent computation trajectories on MNLI and later support causal
early exit.

## Research contract

The active work has three ordered goals:

1. Train a strong MNLI backbone whose recurrent loops retain useful
   sample-dependent trajectories.
2. After such a backbone exists, train an information-rich offline trajectory
   teacher over complete trajectories.
3. Distill that teacher into a causal, one-directional hazard-GRU controller and
   measure the real accuracy--compute frontier.

The offline trajectory teacher is not an inference controller. It sees future
states deliberately and cannot itself provide compute savings. No trajectory
teacher or controller-training path is currently active in the repository; it
will be introduced only after the backbone is strong enough to justify it.

Loop selection used during backbone training must not depend on the gold label
or on a teacher's predicted class distribution. A previous experiment did so
and learned a degenerate depth-to-class code: recurrent depths specialized as
entailment, contradiction, and neutral predictors. Its label-aware oracle reached
99.38%, but its shuffled-label oracle also reached 95.99%, while final matched
accuracy fell to 37.39%. This establishes a permanent control requirement:
oracle lift over shuffled labels, fixed-loop accuracy, per-loop class
histograms, and confidence must all remain healthy.

External semantic distillation may augment the anchored backbone objective, but
it must not replace the final-loop anchor or act as a halting or loop-selection
signal. A discarded unanchored continuation run produced another class cycle:
97.35% label-aware oracle but 87.65% shuffled-label oracle and only 48.12%
final matched accuracy.

Anchored logit distillation avoided that collapse but did not establish a clear
gain over CE-only training. Its six-epoch run reached 66.17% matched accuracy
while opening a 6.87-point sampled-train gap; validation cross-entropy was best
at epoch 3 and worsened thereafter. The active experiment therefore returns to
CE-only training and tests pretrained lexical initialization instead of another
halting or logit-routing objective.

## Model

Let $E\in\mathbb{R}^{n\times d}$ denote token embeddings and $H_t$ the state
entering loop $t$. For $t>0$, the original input is injected by

$$\widetilde H_t = H_t + a\,g\odot \operatorname{MHA}(Q=H_t, K=E, V=E; M),$$

where $g\in\mathbb{R}^d$ is a learned zero-initialized gate and $M$ is the
key-padding mask. Two shared Transformer blocks update the recurrent state. A
rotary loop-time representation places each recurrent step in a distinct
internal coordinate frame.

Input word and premise/hypothesis segment embeddings are initialized from the
pinned `google-bert/bert-base-uncased` checkpoint. A reproducible rank-256 PCA
basis is fitted to its 768-dimensional word embeddings; both word and segment
embeddings are projected through that shared basis. BERT absolute position
embeddings are not imported because this model retains rotary position encoding.

The current static residual coefficient is

$$a=\frac{1}{6\sqrt{2}}\approx0.11785.$$

## Active experiments

### Anchored backbone training

The active six-epoch backbone configuration is `conf/stage/backbone.yaml`. Its
primary objective is unchanged:

$$L_{\mathrm{CE}} = \operatorname{CE}(z_6,y)
+ 0.2\operatorname{CE}(z_s,y),\qquad s\sim\operatorname{Uniform}\{3,4,5\}.$$

The last loop therefore remains the anchor, while one earlier loop per sample
receives weaker direct supervision. External NLI distillation is disabled in
the active configuration so that this run isolates pretrained input
initialization. The first fresh training run loads the pinned BERT embedding
source, caches its projected matrices under `data/pretrained-embedding-cache`,
and records all source and PCA settings in the artifact manifest. Evaluation
and checkpoint resume use saved student weights and do not repeat PCA.

```bash
pixi run -e fallback looped-transformer --config-dir conf train
```

### External semantic distillation

The same training run can add frozen NLI-teacher KL at the final loop and at
the same sampled earlier loop:

$$L = L_{\mathrm{CE}} + \lambda T^2\left[
D_{\mathrm{KL}}(q_T\Vert p_{6,T})
+0.2D_{\mathrm{KL}}(q_T\Vert p_{s,T})\right].$$

The optional external model is `cross-encoder/nli-deberta-v3-base`. Set
`training.stage.loss_options.distillation_scale` above zero to enable this
term. At the active zero weight, the external NLI tokenizer and model are not
loaded and teacher inference is skipped entirely.

Runs with NLI logit distillation use external pretrained and task-finetuned
knowledge and must be described accordingly. The active CE-only run still uses
external self-supervised BERT input embeddings and must not be described as
MNLI-only learning from scratch.

## Evaluation

```bash
pixi run -e fallback looped-transformer --config-dir conf eval \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --data-file data/mnli_matched.tsv \
  --batch-size 64 \
  --output-dir outputs/evaluation \
  --summary-only
```

Evaluation reports:

- final, mean, and per-loop accuracy and cross-entropy;
- true-label probability and prediction confidence by loop;
- confidence split between correct and incorrect predictions;
- adjacent prediction changes, rescues, and harms;
- best fixed loop and label-aware per-sample oracle;
- repeated shuffled-label oracle and genuine oracle lift;
- per-loop preferred-depth and class histograms.

A checkpoint is not useful merely because its label-aware oracle is high.
Shuffled-oracle performance and class histograms must rule out label-to-depth
coding, and fixed-loop task performance must remain competitive.

## Reproducibility

The project uses `pixi` and Python 3.12 or newer.

```powershell
pixi run test
pixi run typecheck
pixi run format
```

Training artifacts record the resolved model, stage, runtime configuration, and
resume-checkpoint fingerprint. Use a distinct output directory for each run.

## Project structure

- `looped_transformer.core`: training, evaluation, checkpointing, and reusable
  PyTorch infrastructure.
- `looped_transformer.domains.mnli`: MNLI model, data, losses, stages, and
  trajectory diagnostics.
- `looped_transformer.app`: configuration, dependency injection, and CLI.

## Scope

The eventual comparisons are fixed-depth versus adaptive-depth execution and
ablations of cross-attention input injection and rotary loop-time encoding.
Attention weights alone are not treated as explanations; analysis must include
prediction trajectories, representation measurements, and causal interventions.

## Prior work

- Dehghani et al., [Universal Transformers](https://arxiv.org/abs/1807.03819),
  2018.
- Banino et al., [PonderNet](https://arxiv.org/abs/2107.05407), 2021.
- Giannou et al., [Looped Transformers as Programmable Computers](https://proceedings.mlr.press/v202/giannou23a.html),
  2023.
- Yang et al., [Looped Transformers are Better at Learning Learning Algorithms](https://openreview.net/forum?id=HHbRxoDTxE),
  2024.
