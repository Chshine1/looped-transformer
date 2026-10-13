# Architecture

The current static warmup uses `a = multiplier / (expected_iterations * sqrt(2))`
on every transformer residual branch and on gated input injection. For six loops,
`a = 0.11785113` at every depth, with no trainable gamma. The equations below
document the historical `geometric` mode retained for old checkpoints; in static
mode, replace the injection gate `g` by `a*g` and each `a_t` by `a`. See
[Static warmup](static_warmup.md) for the experiment and reference.

## Sequence rotation

Each attention head has even dimension (d_h). For position (s) and pair
index (j\in\{0,\ldots,d_h/2-1\}), define

\[
\omega_j=100^{-2j/d_h},\qquad \theta_{s,j}=s\omega_j.
\]

For (x=[x^{(1)},x^{(2)}]\in\mathbb{R}^{d_h}), with two contiguous halves,

\[
R_s(x)=
\begin{bmatrix}
x^{(1)}\cos\theta_s-x^{(2)}\sin\theta_s\\
x^{(2)}\cos\theta_s+x^{(1)}\sin\theta_s
\end{bmatrix}.
\]

The transformation is applied independently to queries and keys before their
dot product. Values are not rotated. Precomputed sine and cosine tables are
non-persistent module buffers, so they follow module device and dtype changes
without enlarging checkpoints.

## Cross-attention input injection

Let (E=\operatorname{Embed}(X)), (H_0=\operatorname{Dropout}(E)), and (M)
be the token-validity mask. At loop (t>0),

\[
I_t=\operatorname{MHA}(Q=H_t,K=E,V=E;M),\qquad
\widetilde H_t=H_t+g\odot I_t.
\]

The current recurrent state therefore selects information from the unchanged
input representation. The learned channel-wise gate (g) is initialized to
zero. Initially, the model is exactly injection-free; the first nonzero
gradient is applied to (g), after which gradients reach the cross-attention
parameters through the opened gate.

## Shared recurrent transition

Let (B_1,B_2) be pre-normalized Transformer blocks sharing their parameters
across loop index. Define

\[
\gamma=\sigma(\ell_\gamma),\qquad a_t=\gamma^t/2.
\]

Each attention and feed-forward residual branch inside (B_1) and (B_2) is
multiplied by (a_t). The transition is

\[
Z_t=\operatorname{RMSNorm}\left(B_2(B_1(\widetilde H_t;a_t);a_t)\right).
\]

## Rotary loop time

For feature-pair index (j\in\{0,\ldots,d/2-1\}), define

\[
\nu_j=5.1^{-2j/d}.
\]

The recurrent state is stored in a frame advanced by one fixed rotation:

\[
H_{t+1}=R_{\nu}(Z_t).
\]

The state used for prediction and halting is mapped back by the cumulative
phase:

\[
Y_t=R_{-(t+1)\nu}(Z_t).
\]

If the shared transition were the identity and input injection were disabled,
(Y_t) would be constant across loops. Consequently, changes in the canonical
state measure the interaction between the shared nonlinear transition and the
loop-dependent internal frame, rather than the rotation alone.

## Prediction and halting

Masked mean pooling gives

\[
u_t=\frac{\sum_s M_sY_{t,s}}{\max(1,\sum_s M_s)},\qquad
z_t=W_c\operatorname{RMSNorm}(u_t)+b_c.
\]

The halting cell receives `stop_gradient(Y_t)`, updates a token-wise GRU state,
and produces a scalar hazard (\lambda_t\in(0,1)). The induced stopping
distribution is

\[
p_t=\lambda_t\prod_{k<t}(1-\lambda_k),
\]

with the final hazard set to one during training so that
(\sum_t p_t=1). Detaching (Y_t) prevents the halting objective from changing
the backbone through the halting-cell input; the task loss can still change the
backbone through (z_t).
