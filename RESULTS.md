# Results so far

All numbers: FineWeb-Edu, GPT-2 BPE, `arch=modern` (RMSNorm + RoPE + SwiGLU + QK-Norm,
bias-free, pre-LN), 122,880 tokens/iter, 2,000 iters, single Kaggle T4 unless stated.
W&B: `hooshaya-ucl/residual-rewiring`.

## Read this first: seed noise is 0.022

Two runs of the **identical** config (L12 baseline, flat LR, 2,000 iters):

| seed | val loss |
|---|---|
| 1337 | 3.9191 |
| 1338 | 3.8968 |
| **spread** | **0.0223** |

Every architectural effect measured below is **≤ that spread**. Nothing here is
established. A new experiment targeting a ~0.02 effect with one run per arm cannot
resolve anything — it will produce a number that reads like a result.

The comparisons *are* paired (same seed, same init, identical eval batches, and zero-init
gates make step 0 identical), so a paired delta may be far more reproducible than 0.022
suggests. That is untested: the run that would have shown it was cancelled. **The cheapest
decisive experiment is `dense` at seed 1338**, since the seed-1338 baseline already exists.

## Constant-LR sweep

Group `finewebedu-flat-s1337`, L12, `lr_schedule=constant` at 6e-4. Deltas vs baseline.

| variant | val | Δ | note |
|---|---|---|---|
| dense-gated | 3.8943 | −0.0248 | also −0.0266 under cosine — replicates across schedules |
| unet-sum (fixed plain sums) | 3.9068 | −0.0123 | was exactly +0.0000 under cosine |
| baseline + α, no warmup | 3.9108 | −0.0083 | was **+0.0765** with warmup + anneal |
| unet-gated | 3.9161 | −0.0029 | |
| baseline | 3.9191 | — | |
| unet-sum + α, no warmup | 3.9354 | +0.0163 | was **+0.2310** with warmup + anneal |

Two things this changed:

- **Flat LR beat cosine outright** (3.9191 vs 3.9849). At 2,000 iters the anneal spends its
  last third at a low LR making little progress. Flat runs are *not* converged — baseline
  was still dropping 0.072 per 200 iters at the end.
- **The α (block-scale) variants were handicapped by warmup + anneal, not broken.** Both
  improved enormously once LR was held flat and warmup removed, which is what ReZero
  predicts: zero-init block scales remove the need for warmup.

## Shallow vs deep: compare on FLOPs, and the answer is no

Never compare depths on **steps** (hands the deeper model ~45% more compute per step) or on
**wall-clock** (Kaggle VM speed varied ~18% between kernels). L6 costs 207.9M FLOPs/token
vs L12's 300.1M — only 31% less, because the tied embedding/lm_head is depth-independent.

| training FLOPs | L6 dense | L12 dense | L6 advantage |
|---|---|---|---|
| 1.5e16 | 4.7029 | 5.0331 | −0.330 |
| 4.5e16 | 4.0835 | 4.1133 | −0.030 |
| 7.16e16 | 3.9329 | 3.9006 | **+0.032 (reversed)** |

L6 leads at low compute and **loses once its compute actually matches L12's**
(same-endpoint run at 2,800 iters = 1.36 epochs, the only run here that repeats data). An
earlier "shallow wins" reading came from interpolating at the budget where L6 happened to
stop — a mistake worth not repeating.

Also: at matched FLOPs, L6-dense vs L12-baseline is +0.0070, inside seed noise. And the L6
advantage that did exist was almost all depth, not skips — L6 dense vs L6 baseline was only
−0.0049, against −0.0248 at L12. **Dense's benefit scales with depth, so it cannot buy
depth back.**

## The gates are alive, and their structure is interpretable

Not a dead-backprop null: all 66 dense edges and all 6 unet edges had nonzero Adam second
moments for all 2,000 steps. What dense learned (`analyse_gates.py` renders it):

- The **`t0` embedding column dominates** — 63% of total gate magnitude — and **flips sign
  with depth**: +0.49 into B1, −0.20 into B9. Early blocks amplify token identity, deep
  blocks subtract it off.
- **Every block from B5 on applies a negative one-step-back term**, which algebraically is
  `0.83·hᵢ + 0.17·Δ` — gain control and a high-pass filter fused into one scalar.
- The **interior is dead** (t2–t6 at depth ≈ 0). 86% of gate energy sits in 30 of 66 edges.
- unet independently agrees on the sign and magnitude of `B11←t0` (−0.156 vs dense's
  −0.146), which is the strongest evidence any of this structure is real.

Caveat: gate *magnitude* is not *contribution*. The honest quantity is
`‖gᵢⱼ·hⱼ‖ / ‖hᵢ‖`, since residual-stream norms grow with depth. Unmeasured.

## Metrics that don't mean what they look like

- **`alpha/sum_abs` does not measure "computation switched on."** `baseline+α` reached only
  2.01 of 24 yet matched baseline. The block computes `α·f(x)`, so f's weights can grow to
  compensate for a small α.
- **Wall-clock is only comparable within a single Kaggle kernel.** Dense costs +4.7%
  wall-clock but only +0.05% FLOPs — its real price is memory traffic (12 live activation
  tensors retained for backward), which the FLOP counter cannot see. So FLOPs flatters
  dense.

## Open next steps

1. `dense` at seed 1338 — one run, gives the paired delta at a second seed.
2. 3 seeds × {baseline, dense} for error bars on the one candidate effect.
3. Sparse topologies the gate matrix actually pointed at: embedding-tap-only (11 edges,
   ~1% overhead vs dense's 4.7%) and one-step-back-only.
4. Richer gate forms — per-channel (25k params, ~0 extra FLOPs) or skips as extra block
   *inputs* rather than summands, which needs no gate at all.
5. A learned per-block gate given a **V-shaped** depth-dependent init (large near input and
   output, zero at the middle) so the middle wakes last. Requires a block-level uniform-zero
   control, not the existing sublayer-α runs. Related published work is all monotone in
   depth (DS-Init, DeepNorm, stochastic depth); check
   [Progressive Residual Warmup](https://arxiv.org/pdf/2603.05369) before assuming novelty.
