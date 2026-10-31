# Looped Transformer for Latent Reasoning Depth

This repository studies whether a small recurrent-depth Transformer can learn
useful, sample-dependent computation trajectories on corrected CLUTRR v1.3
relational classification.

## Research contract

The active work has three ordered goals:

1. Train a strong classification backbone whose recurrent loops remain useful
   and whose trajectories vary with independently measured reasoning depth.
2. Once such a backbone exists, train an information-rich bidirectional GRU
   teacher over complete, frozen trajectories.
3. Distill that teacher into a causal, one-directional hazard-GRU controller and
   measure the real accuracy--compute frontier.

The bidirectional GRU is an offline full-trajectory teacher, not an inference
controller. It deliberately sees future states and cannot save computation. The
eventual deployed controller is causal and is trained only after the backbone
and offline teacher are demonstrably strong.

No loop selector used during backbone training may depend on gold labels or a
teacher's predicted class distribution. Such feedback can leak a depth-to-class
code into the backbone: label-aware oracle accuracy can look excellent even
when shuffled-label oracle accuracy is also high and fixed-loop performance
collapses. A supervised teacher must therefore remain a post-backbone
inference-policy estimator unless a genuine bilevel/meta teaching objective is
introduced.

For CLUTRR, the reference path length is evaluation metadata, not a training or
routing target. A relationship between path length and useful loop depth is
evidence to investigate, not permission to feed the path length or label-aware
oracle back into the backbone.

## Why corrected CLUTRR

Each example is a static classification problem

$$x=(S,a,b),\qquad y\in\mathcal R,$$

where $S$ is a short family story, $(a,b)$ is the query pair, and
$\mathcal R$ contains the 18 gendered kinship targets emitted by the v1.3 GLC
rule base. The relation store also defines spouse and sibling-in-law words, but
those four labels are never emitted as prediction targets. There is no proof or token
generation. The reference resolution path gives an external reasoning length
$k(x)$ without exposing intermediate reasoning states to the model.

The original crowdsourced AMT templates contain incorrect kinship statements,
especially entity-role reversals. CLUTRR v1.3 adds model-screened `is_correct`
flags and a generator option that excludes flagged templates. This is filtered,
not fully manually re-annotated, data. The active provider therefore accepts
only normalized files with a manifest proving all of the following:

- generator version `1.3.0` and an exact source commit;
- AMT templates with `template_amt.ignore_incorrect: true`;
- project-fixed label ordering and file fingerprints;
- normalization of the v1.3 relation-store typo `neice` to `niece`.

The convenient Hugging Face `CLUTRR/v1` dataset is the older release and is not
accepted as corrected v1.3 data.

Two alternatives remain scientifically useful but are not active code paths:

- Corrected StepGame is a strong nine-class follow-up if 18-way CLUTRR proves
  capacity-limited. It has controlled 1--10 hop spatial problems, and its
  original study directly measures recurrent-layer count versus hop count.
- RuleTaker is easier and has clean deduction depth, but its explicit facts and
  rules make the intended algorithm less latent than in CLUTRR.

Keeping one active task avoids turning benchmark choice into another uncontrolled
sweep. StepGame should be added only if CLUTRR results establish a concrete
capacity or data-quality limitation.

## Data preparation

CLUTRR v1.3 currently lives on the official repository's `develop` branch. The
verified reference commit used while implementing this path is
`4f65217eee495b7e7ef6262f6916487e3e4d62c7`.

Clone that revision with submodules and generate a clean logical pool with these
settings:

```text
TEMPLATE_TYPE=amt
NOISE=false
TRAIN_DESCRIPTOR_LENGTHS='2,3'
VAL_DESCRIPTOR_LENGTHS='2,3'
TEST_DESCRIPTOR_LENGTHS='4,5,6'
MAX_PATH_LEN=7
```

Run the official GLC and CLUTRR language generators with the isolated
`clutrr-data` Pixi environment. This environment deliberately excludes the
project's Torch/CUDA training dependencies. From the CLUTRR checkout:

```bash
pixi run -e clutrr-data --manifest-path /path/to/looped-transformer/pyproject.toml \
  python glc/glc.py save_loc=graphs-strict \
  rule_store=glc/rule_bases/clutrr world_prefix=rule \
  num_train_graphs=180000 num_valid_graphs=18000 num_test_graphs=18000 \
  train_descriptor_lengths=\'2,3\' val_descriptor_lengths=\'2,3\' \
  test_descriptor_lengths=\'4,5,6\' add_noise=false \
  noise_policy=dangling max_path_len=7 num_steps=0
```

The GLC uniqueness search does not directly yield useful 60k/6k/6k splits: at
this rule-base size it saturates early, and its descriptor-disjoint validation
split can be tiny. The active preparation therefore filters invalid paths and
samples the depth-2--3 GLC training pool with replacement, balanced by path depth
and ungendered target. IID validation uses an independently rendered subsample
of that same short-composition pool for checkpoint selection. Test uses only the
GLC test pool at unseen depths 4--6. The balancing and preparation steps enforce
length-disjoint train/test descriptors. They then use the official renderer to
create independently seeded names and corrected AMT template realizations. Run
the balancing step on freshly generated `rule_0` files:

```bash
pixi run python scripts/balance_clutrr_v1_3_graphs.py \
  --graph-dir /path/to/generated/rule_0
```

In `clutrr/config.yaml`, keep:

```yaml
template_type: amt
template_amt:
  ignore_incorrect: true
```

Next run the official `clutrr/generator_glc.py` language layer for 60,000
training, 6,000 validation, and 6,000 test examples. It overwrites the balanced
logical files with rendered records. Normalize them:

```bash
pixi run -e clutrr-data --manifest-path /path/to/looped-transformer/pyproject.toml \
  python -m clutrr.generator_glc data_loc=graphs-strict \
  num_train=60000 num_valid=6000 num_test=6000 template_type=amt
```

Then, from this repository:

```bash
pixi run python scripts/prepare_clutrr_v1_3.py \
  --input-dir /path/to/generated/rule_0 \
  --generator-config /path/to/clutrr/clutrr/config.yaml \
  --source-revision 4f65217eee495b7e7ef6262f6916487e3e4d62c7 \
  --balanced-graph-realizations \
  --output-dir data/clutrr-v1.3
```

Preparation makes query direction explicit as
`[second] is the [MASK] of [first].`, matching the v1.3 generator's
`target_gender` convention. The emitted manifest records depth counts, unique
descriptor counts, and a fingerprint of each descriptor set. Existing data
made by the earlier pooled sampler is intentionally rejected and must not be
used for the active experiment.

## Model and objective

The roughly 11M-parameter backbone uses BERT's 768-dimensional word and segment
embeddings projected into the model's 256-dimensional input space with a
reproducible PCA basis. The remaining recurrent Transformer is trained for
CLUTRR rather than imported from BERT.

Let $H_t$ be the recurrent state at loop $t$. Two shared Transformer blocks and
rotary loop-time coordinates update it for six loops. Classification uses masked
mean pooling at every loop. The active loss is

$$L=\operatorname{CE}(z_6,y)
+0.2\operatorname{CE}(z_s,y),\qquad
s\sim\operatorname{Uniform}\{3,4,5\}.$$

The final loop remains the anchor. The sampled earlier-loop term supplies weak
intermediate supervision but does not select a depth or use reference path
length.

Train the active CLUTRR configuration with:

```bash
pixi run -e fallback looped-transformer --config-dir conf train
```

## Evaluation

```bash
pixi run -e fallback looped-transformer --config-dir conf eval \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --data-file data/clutrr-v1.3/test.jsonl \
  --batch-size 64 \
  --output-dir outputs/evaluation \
  --summary-only
```

In addition to final, per-loop, oracle, shuffled-label-oracle, confidence, class
histogram, rescue, and harm metrics, CLUTRR evaluation reports for each reference
path length $k$:

$$A_{t,k}=\Pr(\arg\max z_t=y\mid k(x)=k),$$

per-loop cross-entropy, and the first stably correct loop

$$\tau(x)=\min\{t:\hat y_j(x)=y\ \text{for all}\ j\ge t\}.$$

The last bin of each stable-loop histogram means that the example never became
stably correct. A useful trajectory should show later gains on longer paths
without depth-to-class specialization. Use validation depths 2--3 for checkpoint
selection and report test depths 4--6 separately as length extrapolation; do not
use aggregate test accuracy as the main result. Required controls include:

- label and prediction histograms, also inspected within each path length;
- real path lengths versus shuffled path-length assignments;
- held-out longer paths and held-out relation compositions;
- distractor facts and length-matched controls;
- fixed-loop performance and oracle lift over shuffled labels.

Correlation between $k$ and $\tau$ alone does not prove a latent chain of
thought. Later analysis must intervene on supporting versus irrelevant facts and
inspect whether trajectory changes follow the reference relation path.

The superseded depth-2--4, clean, pooled-sampler run reached 97.95% validation
accuracy at epoch 11. This establishes that the model can fit the clean task,
but it is not evidence of length extrapolation: depth 4 was in training, every
story contained only supporting-path facts, and the sampler did not preserve
GLC split-local descriptor separation. The active depth-2--3 experiment replaces
that protocol rather than extending it.
An audit found 154 unique relation-sequence descriptors in both its train and
validation splits, with all 154 shared; train and test also shared 153. Its
validation score is therefore an interpolation/optimization result, not a
compositional-generalization result.

The replacement depth-2--3 run selected epoch 12 from validation epochs
8--12 by final-loop raw cross-entropy. It reached 99.73% IID validation
accuracy, but only 37.90% on the length-disjoint depth-4--6 test set: 46.20% at
depth 4, 42.50% at depth 5, and 25.00% at depth 6. Recurrent loops improved
aggregate test accuracy from 33.97% at loop 1 to 37.90% at loop 6, while raw
cross-entropy worsened from 2.317 to 2.772 as incorrect predictions became more
confident. The label-aware oracle reached 40.13%, versus 7.42% for the shuffled
label oracle, so the earlier depth-to-class degeneracy is absent but adaptive
headroom is small. Depth 4 contains only 12 of the 18 target classes permitted
by the generated rule compositions, so accuracy differences across depths are
not a pure length comparison; a future label-matched control is required.

## Reproducibility

The project uses `pixi` and Python 3.12 or newer.

```powershell
pixi run test
pixi run typecheck
pixi run python -m ruff check .
pixi run format
```

Training artifacts record resolved model, stage, runtime, data fingerprints, and
resume-checkpoint provenance. CLUTRR itself is CC BY-NC 4.0; check that this is
appropriate before distributing derived data or checkpoints.

## Project structure

- `looped_transformer.core`: reusable training, evaluation, checkpointing, and
  artifact infrastructure.
- `looped_transformer.domains.text_classification`: shared looped backbone,
  tokenizer, loss, pretrained initialization, and diagnostics.
- `looped_transformer.domains.clutrr`: corrected data contract, provider,
  collator, labels, and training-stage plugin.
- `looped_transformer.domains.trajectory_readout`: frozen-trajectory readout
  infrastructure, to be used after a strong backbone exists.

## Prior work

- Sinha et al., [CLUTRR](https://aclanthology.org/D19-1458/), 2019.
- Clark et al., [RuleTaker](https://arxiv.org/abs/2002.05867), 2020.
- Shi et al., [StepGame](https://doi.org/10.1609/aaai.v36i10.21383), 2022.
- Li et al., [Rectified StepGame](https://doi.org/10.1609/aaai.v38i17.29811),
  2024.
- Dehghani et al., [Universal Transformers](https://arxiv.org/abs/1807.03819),
  2018.
- Banino et al., [PonderNet](https://arxiv.org/abs/2107.05407), 2021.
