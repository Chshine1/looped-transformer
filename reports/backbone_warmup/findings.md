# What the backbone warmup checkpoints show

The principal finding is weak additional task benefit after depth 2–3, despite
continued hidden-state movement. The clearest architectural candidate is the
imbalance between exponentially decaying transformer updates and unscaled input
injection. The measurements do not establish that the shared classifier or mean
CE is the primary cause of the performance ceiling.

## Scope and reproducibility

All four warmup checkpoints were evaluated on all 9,815 matched MNLI validation
examples and a fixed random sample of 4,096 training examples. The final diagnostic
run uses 512 randomly selected training examples for gradients and 2,048 randomly
selected validation examples for interventions. Dropout is disabled, every depth
is evaluated, and arithmetic is FP32 without TF32. No model weights or training
configuration were changed.

The machine-readable results are in [report.json](report.json). Collection methods,
reproduction commands and limitations are in
[warmup_diagnostics.md](../../docs/warmup_diagnostics.md). Checkpoint manifests do
not preserve the historical training source; the collector records hashes of the
current implementation and validates the architecture and data against the run.
The checkpoint `best_accuracy` field is not that checkpoint's evaluation result:
the runner saves epoch checkpoints before evaluating and updating that field.

## Accuracy improves over epochs, but mostly saturates early within a forward

| Warmup epoch | Depth 1 accuracy | Depth 2 | Depth 3 | Depth 6 | Any depth correct, label oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51.04% | 52.94% | 53.74% | 53.89% | 64.21% |
| 2 | 55.63% | 58.96% | 59.22% | 59.15% | 67.68% |
| 3 | 57.89% | 61.47% | 61.60% | 61.45% | 69.04% |
| 4 | 59.30% | 62.12% | 62.23% | 62.40% | 69.51% |

At epoch 4, CE by depth is 0.88330, 0.83871, 0.83670, 0.83664,
0.83649, 0.83593. Thus depth 2 supplies most of the improvement over depth 1;
depths 3–6 add little on average. The sampled training accuracy at depth 6 is
65.14%, versus 62.40% on validation. This is not the pattern of near-perfect
training fit with a large validation gap, although one sampled training evaluation
cannot determine whether capacity, optimization, data or regularization dominates.

Flat average CE does not mean every example stops changing. At the last transition
in epoch 4, 27.29% of examples improve by more than 0.01 CE, and 38.61% worsen by
more than 0.01. Predictions flip on 1.09%; 0.469% become correct and 0.428% become
incorrect. These effects nearly cancel in aggregate accuracy. Confidence can move
without changing the predicted class, and a minority of larger improvements can
outweigh a majority of smaller regressions in mean CE.

The label oracle's 69.51% accuracy is evidence of complementary predictions across
depths, not evidence that a usable stopping rule already exists. Its 7.10-point
gap over depth 6 requires choosing using the correct answer. It also spans all six
depths; it must not be attributed solely to the last three depths. The gap is
smaller than epoch 1's 10.32 points, consistent with increasing prediction
agreement during training.

## Hidden trajectories survive the flat CE region

At epoch 4 the exposed token-state changes for depths 3→4, 4→5 and 5→6 are
23.86%, 22.61% and 22.56% of the preceding state norm. Normalized pooled feature
changes are 17.15%, 16.65% and 16.33%. This rules out a literal fixed point of the
measured hidden state at those depths.

Only 0.682%, 0.403% and 0.551% of the respective normalized pooled update energy
lies in the classifier's class-contrast row space. With three classes, that space
has at most two dimensions out of 256. For reference, isotropic movement would put
about 0.781% there. Most of the measured movement is invisible to the current
classifier, but this does not prove a useful internal reasoning workspace: input
injection, normalization and changing internal frames can also produce activity
that does not improve the task.

## A concrete imbalance in the recurrent update

Gamma starts at 0.75 and is 0.5756, 0.4730, 0.4580 and 0.4604 in the saved warmup
epochs. At epoch 4, each transformer residual branch is multiplied by
`[0.5000, 0.2302, 0.1060, 0.0488, 0.0225, 0.01035]`. Depth 6 therefore has about
1/48 of depth 1's residual multiplier. Input injection does not use this multiplier.

The actual updates confirm the imbalance, rather than merely suggesting it from
the formula:

| Epoch 4 component | Depth 2 update / incoming norm | Depth 4 | Depth 6 |
| --- | ---: | ---: | ---: |
| Transformer block 1 | 61.64% | 8.87% | 1.67% |
| Transformer block 2 | 61.00% | 7.74% | 1.32% |
| Gated input injection | 38.81% | 29.54% | 27.39% |

The denominators are each component's own incoming state, so these numbers are
not additive shares. Nevertheless, late loops are much more active in injection
than in the transformer blocks. Normalization and internal-frame effects remain
active too. This is a strong candidate explanation for substantial state movement
without much additional task improvement; update size alone does not establish
which operation supplies useful information.

Changing this balance only at inference does not repair the checkpoint. On the
same 2,048 randomly selected validation examples:

| Intervention | Depth 6 CE | Depth 6 accuracy |
| --- | ---: | ---: |
| Original checkpoint | 0.83665 | 62.45% |
| Gamma + 0.05 | 0.83900 | 62.35% |
| Gamma = initial 0.75 | 0.89742 | 59.62% |
| Gamma approximately 1 | 1.09668 | 44.34% |
| Input injection disabled | 0.88313 | 59.42% |

The network has adapted to its learned transition. Injection contributes to its
current performance; removing it is not a fix. Larger residual updates likewise
require training under that schedule. The small gamma change has a small observed
effect and is not a statistically established regression; the large interventions
show substantial sensitivity.

## What coupling to the classifier actually does

The readout is

`u_t = masked_mean(Y_t); x_t = RMSNorm(u_t); logits_t = W x_t + b`.

A linear head does not indiscriminately blend all dimensions. Its learned weights
can ignore directions, and probabilities are unchanged by movements in the
nullspace of the centered weight matrix **in normalized pooled feature space**.
The observations above show that considerable motion can coexist with almost
unchanged predictions.

There is nevertheless real coupling before normalization. RMSNorm divides by the
RMS over all channels. If `r² = mean(u²) + epsilon` and `a` is its learned channel
scale, its Jacobian is

`J = diag(a) [I/r - u uᵀ/(d r³)]`.

A direction that the linear head ignores can therefore still affect logits by
changing the denominator when considered in raw pooled coordinates. Sharing the
readout across depths also asks their useful features to remain decodable in a
common coordinate system. Neither constraint proves that the model cannot retain
a computation subspace, and the recurrent transitions can mix those subspaces.

Mean pooling mixes **token positions**, not feature channels. Zero-mean changes
across tokens disappear from this readout. Its direct loss gradient is distributed
equally across valid token positions before propagating through the recurrent
backbone; total gradients are not equal because the transformer couples tokens.
Pooling provides a compression bottleneck, but it does not automatically separate
computation and classification channels. Adding only another linear layer before
a linear classifier is not, by itself, extra readout expressivity. A projection
before normalization, a nonlinear adapter, or learned pooling changes the question
more meaningfully, and each needs its own controlled comparison.

Frozen-feature refits do not currently implicate head sharing as the main limit.
At epoch 4, the original last-depth accuracy is 62.40%; refitting one shared head
gives 61.34%, and fitting separate heads gives 61.14%. Their last-depth CEs are
0.85162 and 0.86290, versus the original 0.83593. This limited 4,096-example,
300-step probe did not uncover a gain. It does not rule out a stronger readout,
more probe data, different regularization, or different representations learned
when separate heads are trained from the beginning.

## Gradient evidence: localized conflict, no persistent shared-head cancellation

On the same 512-example training subset, the minimum pairwise head-gradient
cosine across depths is positive at every checkpoint: 0.644, 0.678, 0.732 and
0.687. The ratio `norm(mean gradient) / mean(norm gradient)` is approximately
0.95 throughout. Averaging depth losses does not substantially cancel the head's
gradient on this diagnostic sample.

Backbone gradients are less aligned. Their cancellation ratios are 0.819, 0.644,
0.803 and 0.855. At epoch 2 specifically, depths 4 versus 6 have cosine -0.327 and
depths 5 versus 6 have cosine -0.379. Thus competing depth objectives do occur,
but the observed conflict is localized to late-depth pairs in that checkpoint.
All backbone depth pairs are positively aligned at epoch 4, with minimum cosine
0.322. It would overstate the evidence to describe persistent gradient opposition
as the cause of the final plateau.

At epoch 4, the current-depth versus summed future-depth gradient cosines at the
recurrent token states are 0.049, 0.294, 0.600, 0.832 and 0.937 for states 1–5.
There is considerable directional difference early, but no negative aggregate
alignment. At state 1, the future contribution's norm is about 14.1 times the
local contribution's norm. Equal scalar loss weights therefore do not imply that
the first state's local classification loss dominates its future computation.
These are aggregate, dropout-off measurements at saved checkpoints, not per-example
or historical optimizer trajectories.

The epoch-4 gamma-logit gradients from depths 2–6 are all negative on this training
subset. Locally, this sample's CE would favor increasing gamma, even though gamma
fell over training. That distinction matters: the saved decay is not evidence that
the present deterministic CE gradient is still pushing it down. Training dropout,
different examples, optimizer history and earlier phases can change the direction.

## Implications of the objectives

Mean CE across depths is deep supervision: it rewards useful early predictions.
It does not explicitly require equal hidden states, equal logits or monotonic
per-example CE. It can nevertheless discourage computations that must temporarily
worsen the current prediction before improving a later one. This is a plausible
training bias, not a causal conclusion from checkpoint snapshots.

The later depth-weighted objective still supervises all depths and adds an explicit
penalty on per-example CE increases. That penalty encourages stable exits, but can
also discourage temporary detours. More weight on later losses does not enlarge
the late transformer updates automatically; late losses also backpropagate through
earlier shared computation.

The controller cannot create new classification information while its backbone is
frozen. At the final warmup checkpoint, the label-informed optimum of
`CE + 0.02 * depth` assigns 8,524 of 9,815 examples (86.85%) to depths 1–3 and only
582 (5.93%) to depths 5–6. The implemented controller targets are softened versions
of this cost, rather than hard argmins, but the incentive toward early exits is
clear. Flat late CE makes later computation expensive without compensating benefit.
The warmup controller itself is frozen and absent from the warmup loss, so it does
not explain warmup trajectory suppression.

These findings support testing residual scheduling and the strength of early and
monotonic supervision before assuming the architecture needs disjoint hidden
subspaces. Controlled retraining is required to establish which change improves
accuracy or learnable adaptive depth.

## Next experiments, in priority order

1. Run a controlled warmup comparison crossing the present residual schedule with
   a schedule that preserves late update size, and mean CE with final-depth CE plus
   an annealed early-depth auxiliary term. Keep initialization, data order, optimizer
   and training budget matched; repeat across seeds. This separates the loss bias
   from the residual-schedule bias instead of changing both and attributing a gain
   to one. Monitor the actual component updates as well as gamma.
2. In depth training, independently compare the monotonic penalty on versus off.
   Assess final accuracy, per-example rescues/regressions and useful early exits;
   neither more varied depth choices nor smoother CE alone is success.
3. If those comparisons leave a readout bottleneck, test a task projection before
   normalization or learned pooling with a parameter-budget control. Distinguish
   a pooling change from a depth-specific-head change. The current frozen-feature
   probes do not provide evidence to prioritize six independent classifiers.
4. Evaluate any controller against fixed-depth baselines and label-oracle bounds,
   reporting accuracy versus actual executed depth. A frozen controller stage can
   exploit existing complementary predictions but cannot create better features.

No later-stage checkpoint evaluation or retraining was performed here. The
implications for the approximately 64% later-stage ceiling follow from the warmup
measurements and objective definitions, not from a demonstrated causal diagnosis
of those later stages.
