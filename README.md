# L2OALM.jl

Julia implementation of **Primal-Dual Learning (PDL)** for parametric constrained optimization, following Park & Van Hentenryck, *"Self-Supervised Primal-Dual Learning for Constrained Optimization"* (AAAI 2023).

[![Build Status](https://github.com/LearningToOptimize/L2OALM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LearningToOptimize/L2OALM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/LearningToOptimize/L2OALM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/LearningToOptimize/L2OALM.jl)

---

## What it does

Given a parametric program

```
min   f(y; θ)
 y
s.t.  g(y; θ) ≤ 0      [inequalities]
      h(y; θ) = 0      [equalities]
      lvar ≤ y ≤ uvar  [bounds]
```

PDL trains two networks jointly — **self-supervised**, no solver in the loop:

| Network | Maps | Role |
|---------|------|------|
| Primal `ŷ(θ; φ)` | parameters → decisions | Produces a near-feasible, near-optimal solution |
| Dual `λ̂(θ; ψ)` | parameters → multipliers | Predicts Lagrange multipliers (μ for ineq, λ for eq) |

Training mimics the Augmented Lagrangian Method (ALM): alternate between minimizing the augmented Lagrangian over `φ` and regressing `ψ` onto the ALM multiplier update.

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/LearningToOptimize/L2OALM.jl")
```

`BatchNLPKernels` is a required dependency pinned to its GitHub `main` branch and is resolved automatically.

---

## Quick start

```julia
using L2OALM, Lux, Optimisers, MLUtils, BatchNLPKernels

# 1. Build a BatchModel wrapping your ExaModel (see test/power.jl for AC-OPF)
bm_train = BNK.BatchModel(model, batch_size, config=BNK.BatchModelConfig(:full))
bm_test  = BNK.BatchModel(model, test_size,  config=BNK.BatchModelConfig(:full))

# 2. Define primal and dual networks
primal_net = Chain(Dense(nθ, 512, relu), Dense(512, 512, relu), Dense(512, nvar),
                   BoundedOutput(lvar, uvar))   # enforces bounds architecturally
dual_net   = Chain(Dense(nθ, 512, relu), Dense(512, 512, relu), Dense(512, ncon))

ps_p, st_p = Lux.setup(rng, primal_net)
ps_d, st_d = Lux.setup(rng, dual_net)

# 3. Configure the ALM method
method = ALMMethod(;
    batch_model         = bm_train,
    num_equal           = num_equal,     # number of equality constraints (tail of constraint vec)
    ρmax                = 1e4,
    max_dual            = 1e6,
    τ                   = 0.8,
    α                   = 2.0,
    use_analytical_dual = true,          # apply ALM update analytically per gradient step (recommended)
    use_dual_learning   = true,          # train the dual network (recommended)
)

trainer = ALMTrainer(primal_net,
                     Training.TrainState(primal_net, ps_p, st_p, Optimisers.Adam(1e-4)),
                     dual_net,
                     Training.TrainState(dual_net,   ps_d, st_d, Optimisers.Adam(1e-3)))

data = DataLoader(Θ_train; batchsize=200, shuffle=true)

# 4. Train
train!(method, trainer, data;
    K             = 100,       # outer ALM iterations
    L_primal      = 2500,      # gradient steps per primal phase
    L_dual        = 2500,      # gradient steps per dual phase
    warmup_epochs = 25000,     # penalty-only warm-start steps before ALM loop
    lr_primal     = 1e-4,
    lr_dual       = 1e-3,
    lr_decay      = 0.99,
)
```

---

## Key components

### `BoundedOutput(lvar, uvar)`

Lux layer that enforces variable bounds architecturally using sigmoid:

```
yᵢ = lvar[i] + (uvar[i] − lvar[i]) · σ(zᵢ)
```

Guarantees `max_bound_violation ≡ 0` by construction. Uses sigmoid (not hardsigmoid) so gradient is always nonzero — hardsigmoid's zero-gradient zone at `z < −3` permanently traps outputs at the lower bound.

### `FixRefBus(nvar, ref_bus_idxs)`

Lux layer that zeroes the reference-bus voltage angle architecturally. GPU-safe — mask lives on the same device as the network.

### `ALMMethod`

Immutable configuration struct. Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `ρmax` | `1e6` | Maximum penalty parameter |
| `max_dual` | `1e6` | Multiplier clip bound |
| `τ` | `0.8` | Violation ratio threshold for ρ update |
| `α` | `10.0` | ρ growth factor |
| `ρ_eq_scale` | `1.0` | Extra penalty multiplier for equalities only |
| `use_analytical_dual` | `true` | Apply ALM dual update per gradient step |
| `use_dual_learning` | `true` | Train the dual network |

**`use_analytical_dual`**: instead of using the dual network output directly, computes analytically-corrected multipliers at every primal gradient step:
```
μ_eff = clamp(μ̂(θ) + ρ·g(ŷ),  0,  max_dual)
λ_eff = clamp(λ̂(θ) + ρ·h(ŷ), −max_dual, max_dual)
```
Eliminates the dual tracking gap (the dual network lags the true multipliers by orders of magnitude at high ρ without this correction).

**`use_dual_learning`**: set `false` to freeze the dual network — useful for ablations where you want to isolate the effect of the analytical correction alone.

### `ALMTrainer`

Mutable state struct holding both networks, their `TrainState`s, the current penalty `ρ`, and a snapshot of the previous dual state used for computing ALM targets.

### `train!(method, trainer, data; ...)`

Top-level training loop:
1. **Warm-start** (`warmup_epochs` gradient steps): penalty-only primal loss at `ρmax` to push the network into a feasible basin before the dual starts producing meaningful multipliers.
2. **K outer iterations**: alternate primal phase (`L_primal` steps) and dual phase (`L_dual` steps); update ρ if violations stagnate; optional `eval_fn` callback with learning-rate decay.

---

## Constraint ordering

Constraints must be ordered as **inequalities first, equalities last**. `num_equal` is the count of equalities at the tail of the constraint vector. See `test/power.jl` for how to build a compliant `ExaModel` for AC-OPF.

---

## AC-OPF benchmark (case57)

Results on `pglib_opf_case57_ieee` (128 variables, 435 constraints: 320 ineq + 115 eq), 5000 held-out test samples:

| Config | max_eq | max_ineq | Notes |
|--------|--------|----------|-------|
| `use_analytical_dual=true`, ρmax=1e4, max_dual=1e6 | **1.165** | 0.000 | Best result |
| `use_analytical_dual=false` (dual network only) | ~1.25 | 0.000 | Tracking gap hurts |
| ρmax=1e6, max_dual=1e6 | 1.741 | 0.000 | Oscillates at high ρ |
| max_dual=1e4 (any ρ schedule) | ~1.80 | 0.000 | Gradient saturates at 1.0/step |

Variable bounds and inequality constraints are satisfied exactly (`max_bound = max_ineq = 0`) in all runs by iter ~10.

---

## Running tests

```bash
# CPU
julia --project=. -e 'using Pkg; Pkg.test()'

# GPU (CUDA)
BNK_TEST_CUDA=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

The test uses `pglib_opf_case14_ieee` (downloaded automatically via the `PGLib` artifact on first run).

---

## Examples

- [`examples/case57_train.jl`](examples/case57_train.jl) — single-phase training on case57; all hyperparameters overridable via `PDL_*` env vars.
- [`examples/case57_train_twophase.jl`](examples/case57_train_twophase.jl) — two-phase training: Phase 1 at fixed high ρ (avoids degenerate pg=0 basin), Phase 2 grows ρ with `ρ_eq_scale`.

---

## Reference

```bibtex
@inproceedings{park2023pdl,
  title     = {Self-Supervised Primal-Dual Learning for Constrained Optimization},
  author    = {Park, Seonho and Van Hentenryck, Pascal},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2023},
  url       = {https://arxiv.org/abs/2208.09046}
}
```
