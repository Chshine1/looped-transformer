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

Training is configuration-driven and produces one artifact for one stage. The
default stage retrains the backbone for six epochs in FP32 with batch size 200.
To continue an interrupted run, put the checkpoint in the composed
configuration rather than passing state to the trainer:

```bash
pixi run looped-transformer --config-dir conf \
  --override training.resume_from=/absolute/path/to/checkpoint-epoch-2.pt \
  train
```

The artifact identity includes the resolved stage configuration, the resume
checkpoint fingerprint, and (for a controller teacher) the backbone source
checkpoint fingerprint. A controller teacher run can therefore be selected
with only the differing Hydra overrides:

```bash
pixi run looped-transformer --config-dir conf \
  --override stage=controller_teacher \
  --override training.stage.source_checkpoint=/absolute/path/to/backbone.pt \
  train
```

On a 24 GiB RTX 4090, controller collection uses backbone batches of 200 but
slices each collected trajectory into teacher microbatches of 40. Five-way
gradient accumulation retains an effective batch size of 200. For six loops,
sequence length 128, and hidden dimension 256, one collected FP32 trajectory is
about 150 MiB; a teacher microbatch is about 30 MiB before activations. The
default one-epoch teacher schedule performs about 1,964 frozen-backbone forwards,
9,818 teacher microsteps, and 1,964 optimizer steps on the full MNLI training
set. In practice, budget roughly one to two backbone-training epochs per teacher
run; the three teachers add approximately 50--100% to the six-epoch backbone
run. Actual wall time depends strongly on tokenization and CUDA kernel throughput.

The controller teacher is a soft, full-trajectory depth teacher. It predicts
each loop's correctness independently with one smoothed BCE objective, avoiding
the false requirement that exactly one loop must be correct. During guided
backbone training, those correctness scores are converted into a moderately
focused distribution, blurred across neighboring loops and mixed with a uniform
floor. Because its bidirectional GRU consumes the complete trajectory, it can
guide later backbone refinement but does not itself provide online early-exit
compute savings.

Run the complete backbone plus epoch-2, epoch-3, and epoch-6 teacher sequence on
Ubuntu with:

```bash
bash scripts/train_backbone_and_teachers.sh
```

For alternating training, first reserve a deterministic 4,000-example slice of
matched validation for cycle selection. The untouched remainder and mismatched
validation are used only for final comparisons:

```powershell
python scripts/split_mnli_cycle_data.py
```

Run the full gated controller/backbone alternation from the chosen (normally
epoch-2) backbone checkpoint:

```bash
bash scripts/run_alternating_training.sh \
  outputs/<run>/<artifact>/backbone_warmup/checkpoint-epoch-2.pt 1
```

Each cycle trains a one-epoch controller with smoothed per-loop correctness BCE,
evaluates it on both sampled training data and `data/mnli_cycle_dev.tsv`, and
rejects controllers that are uninformative, nearly uniform, over-confident, or
worse than the final loop. Controller sigmoid scores are converted to soft loop
weights with moderate temperature sharpening, neighboring-loop smoothing, and a
uniform floor. The guided stage freezes the controller, detaches those weights,
removes the final-loop anchor, and saves a plain backbone-compatible checkpoint.
The new backbone must also preserve cycle-dev accuracy and cross-entropy before
it becomes the source for the next cycle. Final matched and mismatched holdouts
remain untouched by cycle selection.

To run just the guided stage manually:

```powershell
pixi run looped-transformer --config-dir conf `
  --override stage=guided_backbone `
  --override training.stage.source_checkpoint=outputs/<backbone>.pt `
  --override training.stage.controller_checkpoint=outputs/<controller>.pt `
  --override checkpoint.output_dir=outputs/guided-backbone-cycle-1 `
  train
```

## Project structure

The codebase is organized as a modular monolith with a one-way dependency graph:

- `looped_transformer.core` contains domain-independent contracts,
  checkpointing, training/evaluation infrastructure, PyTorch extensions, and
  reusable neural-network components;
- `looped_transformer.domains.mnli` contains the complete MNLI vertical slice;
- `looped_transformer.app` is the configuration, dependency-injection, and CLI
  composition root.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the dependency rules, guidance for
adding another training domain, and when it would be worthwhile to split this
distribution into a multi-project Python workspace.

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
  --data-file data/mnli_matched.tsv --batch-size 64 `
  --output-dir outputs/<run>/evaluation
```

The output directory contains `batches.jsonl` with one structured record per
batch and `summary.json` with the final result and aggregate diagnostics. If
`--output-dir` is omitted, the files are written to an `evaluation` directory
beside the checkpoint.

Use a distinct `--output-dir` for each checkpoint/dataset combination; evaluation
replaces logs in that directory. The default dataset is matched validation.
To compare every warmup checkpoint on sampled training and matched validation
using the configuration recorded in `artifact.json`:

```powershell
python scripts/checkpoint_depth_sweep.py --run-dir outputs/<run>
python scripts/summarize_depth_sweep.py outputs/<run>/depth_analysis
```

The sweep saves aggregate JSON diagnostics and per-example logits/labels under
`depth_analysis`. Existing trajectory caches are reused; use a fresh directory
or remove the corresponding cache if the dataset or evaluation implementation
changes. `--splits train_sample matched mismatched` includes both validation sets.
The additional diagnostics compare all loops with supervised loops 3–6 and
include causal confidence-threshold baselines.

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
