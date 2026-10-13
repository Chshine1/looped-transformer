# Reproducing the warmup investigation

The collector evaluates existing checkpoints without updating their weights or
changing the training implementation. It uses the production forward and temporary
hooks. The root module's training flag requests all depths while child modules stay
in evaluation mode, disabling dropout. Hooks and module modes are restored even on
failure. Gradients are computed with `autograd.grad`; parameter `.grad` buffers and
optimizer state are untouched.

```powershell
$env:HF_HUB_OFFLINE = '1'
pixi run python -m looped_transformer.analysis.warmup --eval-examples 9815 --train-examples 4096 --gradient-examples 512 --intervention-examples 2048 --batch-size 16
```

The default checkpoint directory is resolved from the configuration's artifact
identity. Use `--run`,
`--config`, `--output`, and `--epochs` to select another run or subset. Data and
architecture must agree with the run manifest. Tokenizer assets must already be
cached. The run uses FP32 with TF32 disabled; it does not reproduce stochastic
training-mode loss or mixed-precision rounding exactly.

The implementation is organized as follows:

- `src/looped_transformer/analysis/reports.py`: typed report schemas.
- `src/looped_transformer/analysis/collect.py`: forward collection, trajectory
  summaries, gradient comparisons, and frozen-feature head fitting.
- `src/looped_transformer/analysis/warmup.py`: deterministic sampling, manifest
  checks, checkpoint loading, interventions, and incremental report serialization.
- `tests/test_analysis.py`: production-forward equivalence, sample-weighted
  microbatch gradient equivalence, and restoration of module modes.

Each completed checkpoint is written to `reports/<run-id>/backbone_warmup/report.json`,
with a compact generated `summary.md`. The report records source/checkpoint/data
hashes, exact zero-based data-row selections, configuration, environment, and seed.
JSON rejects nonfinite numbers. Source hashes identify the code used in collection;
the old training checkpoints themselves do not archive their training source.
Do not run two collectors against the same output directory concurrently.

## Measurements and interpretation

CE and accuracy are computed for each depth on identical examples. Adjacent
improvement/regression fractions use a CE threshold of 0.01. CE-change standard
errors use paired per-example differences. Best-depth counts break ties in favor
of the earliest depth; near ties should not be interpreted as robust depth needs.
The cost-optimal depth minimizes `CE + 0.02 * one_based_depth`. Both this and the
any-depth-correct result use ground-truth labels and are oracle diagnostics.

Representation changes use masked exposed token states, pooled states before
normalization, and normalized classifier inputs. Relative changes divide each
example's update norm by its preceding state norm, then average over examples.
Classifier-visible energy projects changes in normalized pooled features onto
the row space of the centered classifier weight matrix. For three classes this
space has at most two dimensions. A small fraction is therefore not surprising
by itself: isotropic motion in 256 dimensions would put about 2/256 of its energy
there. Motion outside this space is not evidence of useful reasoning.

Component updates measure the actual masked output-minus-input norm for each
whole transformer block, including its scaled residual branches. Injection updates
measure the gated attention contribution relative to the incoming state. Block
lists contain all depths; injection lists start at depth 2, where injection begins.
These norms measure activity, not causal task importance.

Parameter gradients are gradients of each depth's sample-mean CE on the same
random training subset, accumulated with example-count weighting. They exclude
the halting cell, which is frozen and irrelevant to the warmup CE objective.
Cosine matrices are grouped by head, transformer backbone, embedding, injection,
and gamma. The all-parameter norm is sensitive to parameterization; a large head
gradient cannot on its own establish that the head is the limiting component.
Cancellation ratio is `norm(mean(g_t)) / mean(norm(g_t))`. Zero-norm cosine entries
are encoded as zero, including the depth-1 injection/gamma gradients.

State-gradient measurements use the token state immediately after recurrent
RMS normalization and before output/internal rotations. They compare the direct
current-depth contribution `grad(L_t / T)` with the sum of contributions from all
future CE terms at that same state. Norms and dot products aggregate over the
recorded examples and token coordinates. The final state's future gradient is
zero by construction. Positive agreement on an aggregate sample does not exclude
conflict for individual examples or at earlier training steps.

Probe comparisons freeze all features and fit either one shared linear head or
one head per depth, initialized from the checkpoint head. Both use 300 full-batch
AdamW steps on 4,096 sampled training examples, learning rate 0.003 and weight
decay 0.01. Classifier RMS normalization remains frozen. Evaluation labels never
enter fitting. These are limited-budget readout probes, not a controlled comparison
of training architectures; separate heads have more parameters and can overfit.

Final-checkpoint interventions use the same randomly selected validation examples:
baseline, gamma increased by 0.05, gamma restored to 0.75, gamma approximately 1,
and input injection disabled. Each restores the original state afterward. These
are inference-time distribution shifts, not substitutes for retraining ablations.

## Related methodological context

For static scaling, gamma is null and its gradient list is empty. The intervention
set instead uses baseline, both residual/injection coefficients multiplied by 0.75
or 1.25, and injection disabled. Schema version 2 records the mode and per-depth
coefficients. Historical reports in `reports/backbone_warmup` are preserved.

Supervising every intermediate prediction is a form of deep supervision; its
intention includes encouraging early discriminative representations. See
[Deeply-Supervised Nets](https://proceedings.mlr.press/v38/lee15a.html).
Gradient disagreement is a useful diagnostic of competing objectives, but it is
not equivalent to all forms of optimization difficulty; see
[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782).
Neither paper establishes the cause of this run's plateau; conclusions must come
from these measurements and subsequent controlled experiments.
