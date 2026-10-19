# Looped Transformer for Natural-Language Inference

This repository studies recurrent-depth computation on MNLI. Two mechanisms are
treated as independent experimental factors:

1. cross-attention input injection from the recurrent state into the original
   token embeddings; and
2. a rotary loop-time representation that places each recurrent step in a
   distinct internal coordinate frame.

The model produces a prediction and a halting hazard at every loop. The initial
experiments test whether these mechanisms improve task learning, create
iteration-specific computation, and support input-dependent stopping.

## Model

Let $E\in\mathbb{R}^{n\times d}$ denote token embeddings and $H_t$ the state
entering loop $t$. For $t>0$, the original input is injected by

$$\widetilde H_t = H_t + a\,g\odot \operatorname{MHA}(Q=H_t, K=E, V=E; M),$$

where $g\in\mathbb{R}^d$ is a learned zero-initialized gate and $M$ is the
key-padding mask. Thus injection is cross-attention, not direct embedding
addition. Two shared Transformer blocks then update the state. The current warmup
uses the same static coefficient for injection and every attention/FFN residual:

$$a=\frac{\lambda}{N_{\rm ref}\sqrt{2}}=\frac{1}{6\sqrt{2}}\approx0.11785.$$

The reference iteration count is a fixed configuration value, not an estimate
from the controller. Static mode has no trainable gamma. The historical geometric
mode remains available for old checkpoints.

## Reproducibility

The project uses `pixi` and Python 3.12 or newer.

```powershell
pixi run test
pixi run typecheck
pixi run train
```

## Evaluation

Evaluation is composed from an inference runner and an evaluation policy. The
built-in `LoopDiagnosticsPolicy` treats the final-loop score as the deployable
warmup metric, while emphasizing whether the backbone exposes useful stopping
opportunities elsewhere in the trajectory. It compares the label-aware
per-sample oracle with both the best fixed loop and a repeated shuffled-label
oracle, and records rescue rates, preferred-loop histograms, prediction changes,
and per-loop class histograms. The oracle measures backbone headroom; it is not
an estimate of an untrained controller's performance.

```powershell
python -m looped_transformer.main --config-dir conf eval `
  --checkpoint outputs/<run>/<checkpoint>.pt `
  --output-dir outputs/<run>/evaluation
```

The output directory contains `batches.jsonl` with one structured record per
batch and `summary.json` with the final result and aggregate diagnostics. If
`--output-dir` is omitted, the files are written to an `evaluation` directory
beside the checkpoint.

## Scope

The initial claim is comparative rather than architectural novelty. Required
comparisons are fixed-depth versus adaptive-depth execution, and ablations of
cross-attention injection and rotary loop-time encoding. Attention weights alone
are not treated as explanations; analysis must include prediction trajectories,
representation measurements, and causal interventions.

## Prior work

- Dehghani et al., [Universal Transformers](https://arxiv.org/abs/1807.03819),
  2018.
- Banino et al., [PonderNet](https://arxiv.org/abs/2107.05407), 2021.
- Giannou et al., [Looped Transformers as Programmable Computers](https://proceedings.mlr.press/v202/giannou23a.html),
  2023.
- Yang et al., [Looped Transformers are Better at Learning Learning Algorithms](https://openreview.net/forum?id=HHbRxoDTxE),
  2024.
