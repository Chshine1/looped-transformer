# Staged Training Plan

## Motivation

The backbone and halting controller solve different optimization problems. A
frozen, untrained controller must not weight the backbone objective, and a
frozen backbone gives the controller a supervised stopping target that is more
informative than a fixed geometric prior. The default run therefore uses four
objectives and two scheduler types.

With the current 392,654-example cleaned training file and batch size 32, one
epoch contains 12,271 optimizer steps. A loss reported at step 1,500 covers
approximately 0.12 epochs. Logging uses a mean over each 50-step window and also reports the
six individual loop cross-entropies; individual minibatch losses are not used
to assess convergence.

## Shared optimization settings

- AdamW: ((\beta_1,\beta_2)=(0.9,0.999)), (\epsilon=10^{-8}).
- Weight decay: (0.01) for matrix parameters; zero for biases, normalization
  scales, gates, and other parameters with fewer than two dimensions.
- Global gradient-norm limit: (1.0).
- Batch size: 32; no gradient accumulation by default.
- CUDA: FP16 autocast, FP32 loss reduction, and TF32 matrix multiplication.

Cosine schedules use a multiplicative floor. If (s) is the post-warmup step,
(S) is the number of post-warmup steps, and (r_{min}=0.1), the multiplier is

\[
r(s)=r_{min}+(1-r_{min})\frac{1+\cos(\pi\min(s,S)/S)}{2}.
\]

The same multiplier is applied to every parameter group, preserving learning-
rate ratios.

## Stage 1: backbone warm-up

Duration: 4 epochs. The complete halting cell is frozen and placed in evaluation
mode; its dropout is therefore disabled.

For loop logits (z_t), label (y), batch size (B), and (T=6),

\[
\mathcal L_{mean}=\frac{1}{BT}\sum_{i=1}^{B}\sum_{t=1}^{T}
\operatorname{CE}(z_{t,i},y_i).
\]

This loss is independent of all hazard values and stopping probabilities. It
provides the simplest direct classification signal to every recurrent depth.

| Parameter group | Peak learning rate |
| --- | ---: |
| embeddings, shared blocks, normalizations, classifier | (5\times10^{-4}) |
| input-injection attention | (5\times10^{-4}) |
| injection gate and residual-decay logit | (10^{-3}) |

Scheduler: 6% linear warm-up from 0.1 of the peak rate, then multiplicative
cosine decay to 0.1 of the peak rate.

Exit conditions:

1. mean training loss is below the random-label value (log 3\approx1.099);
2. matched validation accuracy exceeds the majority baseline;
3. all loop cross-entropies are finite.

Failure of these conditions blocks the later stages. First test overfitting on
128 examples, then learning on a fixed 20,000-example subset, before repeating
the stage on full MNLI.

## Stage 2: depth-structured backbone training

Duration: 8 epochs. The complete halting cell remains frozen in evaluation mode.

Define increasing normalized weights

\[
w_t=\frac{\exp(\kappa(t-1))}{\sum_{j=1}^{T}\exp(\kappa(j-1))},
\qquad \kappa=0.5.
\]

For six loops these are approximately
([0.034,0.056,0.092,0.152,0.251,0.414]). The loss is

\[
\mathcal L_{depth}=
\frac1B\sum_i\sum_t w_t\operatorname{CE}_{t,i}
+\mu\frac{1}{B(T-1)}\sum_{i,t<T}
\max(0,\operatorname{CE}_{t+1,i}
-\operatorname{stopgrad}(\operatorname{CE}_{t,i})),
\]

where (\mu=0.2). The stopped comparison prevents the monotonic term from
reducing itself by making an earlier prediction worse. Increasing task weights
allow later loops to specialize without leaving early predictions completely
unsupervised.

| Parameter group | Peak learning rate |
| --- | ---: |
| embeddings, shared blocks, normalizations, classifier | (2\times10^{-4}) |
| input-injection attention | (2\times10^{-4}) |
| injection gate and residual-decay logit | (3\times10^{-4}) |

Scheduler: 2% linear warm-up from 0.5 of the peak rate, then multiplicative
cosine decay to 0.1 of the peak rate.

Exit conditions:

1. final-loop validation loss is no worse than after Stage 1;
2. mean positive degradation
   (\max(0,\operatorname{CE}_{t+1}-\operatorname{CE}_t)) decreases;
3. at least one later loop improves validation loss over loop one.

The third condition is necessary for input-dependent computation to have a
task-level benefit.

## Stage 3: controller training

Duration: 2 epochs. Only `_halting_cell` is trainable. The backbone, input
injection, loop rotation, classifier, residual decay, and injection gate are
frozen and placed in evaluation mode. Backbone dropout is disabled, making the
oracle losses and controller inputs deterministic for a fixed example. The
halting cell remains in training mode.

For each labelled training example, define the frozen-backbone oracle cost

\[
C_{t,i}=\operatorname{stopgrad}(\operatorname{CE}_{t,i})+c\,t,
\qquad c=0.02,
\]

and the soft target distribution

\[
q_{t,i}=\operatorname{softmax}_t(-C_{t,i}/\tau),
\qquad \tau=0.1.
\]

For predicted stopping probabilities
(p_{t,i}=\lambda_{t,i}\prod_{j<t}(1-\lambda_{j,i})), optimize

\[
\mathcal L_{controller}=-\frac1B\sum_i\sum_t q_{t,i}\log p_{t,i}.
\]

The label is used only to construct training targets. At evaluation, the
controller observes hidden states and hazards, not the label. This formulation
trains the controller to approximate the empirical loss--compute optimum of the
already trained model; it does not force a label-independent geometric stopping
distribution.

Learning rate: (5\times10^{-4}). Scheduler: 5% linear warm-up from 0.1 of the
peak rate, followed by a constant rate. A constant post-warmup rate is used
because this is short supervised calibration from a newly initialized
controller.

Exit conditions:

1. controller cross-entropy decreases;
2. stopping mass is not concentrated entirely on one loop;
3. predicted expected depth is positively associated with oracle depth on the
   validation set.

## Stage 4: joint adjustment

Duration: 4 epochs. No parameters are frozen.

Let (p_t) be the predicted stopping distribution. The joint objective is

\[
\begin{aligned}
\mathcal L_{joint}={}&
\frac1B\sum_i\sum_t p_{t,i}\operatorname{CE}_{t,i}
+c\frac1B\sum_i\sum_t t\,p_{t,i}\\
&+0.1\,\mathcal L_{depth}
+0.1\,\mathcal L_{controller},
\end{aligned}
\]

with (c=0.02), (\tau=0.1), (\kappa=0.5), and (\mu=0.2). The expected
risk trains both classifier paths and controller probabilities; the direct
compute term exposes the accuracy--depth trade-off; the two small stabilizers
retain the structures learned in Stages 2 and 3.

| Parameter group | Peak learning rate |
| --- | ---: |
| embeddings, shared blocks, normalizations, classifier | (5\times10^{-5}) |
| input-injection attention | (10^{-4}) |
| injection gate and residual-decay logit | (10^{-4}) |
| halting cell | (2\times10^{-4}) |

Scheduler: 2% linear warm-up from 0.5 of the peak rate, then multiplicative
cosine decay to 0.1 of the peak rate.

## Measurements and ablations

Every validation records fixed-loop accuracy and cross-entropy, adaptive
accuracy, expected loop count, stopping histogram, prediction changes, state
and update norms, (\gamma), and injection-gate statistics. The first controlled
comparison uses three identical seeds for:

| Variant | Cross-attention injection | Rotary loop time |
| --- | --- | --- |
| base | off | off |
| injection | on | off |
| time | off | on |
| combined | on | on |

The controller is compared against fixed depths, the oracle stopping rule, and
a confidence-threshold rule. Mechanistic measurements include per-loop probes,
linear CKA between canonical states, causal replacement of injection updates,
and inference-time ablation of the injection gate and loop rotation.
