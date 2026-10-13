# Static scaling warmup experiment

The reference is Wang et al., [On the Residual Scaling of Looped Transformers:
Stability and Transferability](https://arxiv.org/html/2606.18524v1), section 4.
It proposes `lambda / (N * sqrt(L))`. We take `L = 2` unique transformer blocks,
`lambda = 1`, and a fixed `N_ref = E[N] = 6` for this full-depth warmup. Thus each
attention and FFN residual branch uses `1 / (6 * sqrt(2)) = 0.11785113`.
The square-root factor is in the denominator; the old extra division by two is
not applied. There is no geometric exponent or trainable gamma in static mode.

This is a paper-inspired baseline, not a stability guarantee for our rotary loop
frames, recurrent normalization, or cross-attention injection. Applying the same
coefficient to injection is an experimental extension. The injection gate remains
zero-initialized and trainable. Injection still starts at the second loop;
the same reference budget is used rather than a separate five-step denominator.

`model.residual_scaling` selects `static` or historical `geometric` behavior.
`expected_iterations` is a fixed reference budget in `[1, max_iterations]`;
omitting it uses `max_iterations`. It is not recomputed from halting probabilities.
`multiplier` must be finite and positive. Existing configurations without this
section retain historical geometric behavior and checkpoint parameter names.

The current `config.yaml` schedules only `backbone_warmup`, for four epochs.
Mean-loop CE, learning rates, the injection gate, normalization, rotation, and
controller freezing are unchanged. Later-stage templates remain available.

Run from a fresh initialization:

```powershell
pixi run train
```

Do not resume a geometric checkpoint into this experiment: static mode does not
contain its learned gamma parameter. Artifact identity includes the scaling
configuration and stage schedule, so the new run has its own directory.

After training, collect the same full-validation diagnostics:

```powershell
$env:HF_HUB_OFFLINE = '1'
pixi run python -m looped_transformer.analysis.warmup --eval-examples 9815 --train-examples 4096 --gradient-examples 512 --intervention-examples 2048
```

The collector defaults to the run identified by `config.yaml` and writes to
`reports/<run-id>/backbone_warmup`, preserving historical reports. Schema version 2
records the scaling mode, actual per-depth residual and injection coefficients,
and null gamma with an empty gamma-gradient list for static models. Injection
update measurements include the new coefficient. Static interventions multiply
both residual and injection scales by 0.75 or 1.25 and restore them afterward.

Compare CE/accuracy trajectories, per-example rescues and regressions, actual
component update sizes, state motion visible to the classifier, and current versus
future state gradients against the historical report. These measurements can show
where an input-dependent update budget may help; changing the controller or
supervision at the same time would obscure that comparison.
