# Residual scaling and warmup regularization

## RoPE precision

Both positional and time RoPE store separate real cosine/sine buffers.
The rotation `(a*cos - b*sin, a*sin + b*cos)` is mathematically equivalent
to complex multiplication without complex dtype restrictions. In particular,
`model.to(dtype=float32)` cannot discard the sine component anymore.
Rotation arithmetic runs outside autocast in float32 (float64 for double
activations), then returns the input dtype. Under normal AMP usage, model
parameters and caches stay float32. Explicitly converting the whole model to
half/bfloat16 also rounds the cached tables, so prefer float32 parameters
with AMP over converting the model itself to half precision.

The config now disables AMP for a float32 baseline. TF32 remains enabled
for eligible CUDA matrix multiplications; disable `allow_tf32` as well when
comparing strict float32 arithmetic. Available GPU memory still determines
the usable microbatch size. RoPE supports float16/bfloat16 activations if
AMP is enabled again. The buffers remain nonpersistent, so this cache change
does not add/remove checkpoint state entries. Previously trained weights
may nevertheless reflect training with incorrect rotations.

There are five learned scalar gains: one for AttnRes and separate attention
and FFN gains in each of the two inner transformer blocks. Each is
`lambda = 2 * sigmoid(logit)`, with `logit = 0` initially. AttnRes therefore
uses `lambda / max_iterations`; the inner branches retain
`lambda / (sqrt(2) * max_iterations)`. Gains are shared across iterations.
They are positive and bounded (floating point saturation can reach 0 or 2),
and initialization preserves the old residual coefficients. These bounds
limit the gain, not the norm of the whole network's update.

The scalar choice preserves RoPE pairs. Independent per-real-channel gains
would introduce another source of noncommutation. Scalar logits receive no
AdamW decay and a lower learning rate than the projection weights.

## Soft time-RoPE constraint

Time RoPE rotates all complex coordinates within a head by the same angle,
with distinct frequencies across heads. Its commuting real linear maps have
zero cross-head blocks and diagonal head blocks of the form
`[[A, -B], [B, A]]`. A and B are arbitrary square matrices: the constraint
does not require individual complex coordinates within a head to be isolated.

For each Q/K weight W, let P(W) be its orthogonal projection onto that
subspace. The penalty is the mean over Q and K of
`||W - P(W)||_F^2 / d_model + ||bias||_2^2 / d_model`.
The bias term matters because an affine map commutes only if its bias is
fixed by the rotation; the current nonzero head angles require zero bias.
Projection is differentiable and used only to measure distance: forward
attention still uses unrestricted weights and biases. Computing the penalty
in float32 avoids squaring small deviations in float16.

This structural distance has the same zero set as commuting with the current
time rotation, without weakening the penalty between nearby head frequencies.
It must be revisited if time frequencies become shared across heads or the
real/imaginary layout changes. It constrains Q/K only, not all of AttnRes.

Warmup adds `time_rope_regularization_scale * penalty` to its task loss.
The YAML starts at 0.01; omitted/zero disables it for older loss configs.
Other training stages do not add this penalty, so later training can move
away from the structure. The coefficient and learning rates are initial
hyperparameters to evaluate, not empirically tuned optima.

## Optimizer groups

Patterns match the full owning module name, with the first match winning.
Specific scale and normalization rules precede broad transformer rules.
Both recurrent blocks use the same rate because both repeat at every depth.
Warmup uses 4e-4 for embeddings and recurrent block matrices, 3e-4 for
AttnRes Q/K, 5e-4 for the classifier, 2e-4 for normalization, and 1e-4 for
gains/history weights. Subsequent stages reduce backbone rates; controller
training freezes the backbone. All scalars, biases, and norm vectors are
exempt from weight decay through the existing dimensionality rule.

The configured stage sequence remains unchanged. All four stage templates
have matching patterns, including templates not currently in that sequence.

AttnRes history is now appended after each loop; previously its history
argument was always empty and the branch returned zero. This intentionally
changes behavior beyond adding gains. New scale parameters also mean old
checkpoints will not load strictly without migration; optimizer states with
the old groups should not be resumed with the new groups.
