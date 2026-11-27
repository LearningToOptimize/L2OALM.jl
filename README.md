<div align="center">

# L2OALM.jl

<p><sub><b>PRIMAL PROXY · DUAL PROXY · AUGMENTED LAGRANGIAN</b></sub></p>

[![CI](https://github.com/LearningToOptimize/L2OALM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LearningToOptimize/L2OALM.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://LearningToOptimize.github.io/L2OALM.jl/dev/) [![Codecov](https://codecov.io/gh/LearningToOptimize/L2OALM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/LearningToOptimize/L2OALM.jl) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

</div>

<p align="center">
  <picture>
    <source media="(max-width: 640px)" srcset="docs/src/assets/diagrams/l2oalm--primal-dual-compact.svg">
    <img src="docs/src/assets/diagrams/l2oalm--primal-dual.svg" width="760" alt="Self-supervised primal-dual learning in three panels. Primal phase, dual proxy frozen: the parameters p enter the primal proxy, which outputs x-hat; with lambda equal to the frozen dual proxy output, the primal loss L_p (B2) is s f of x-hat plus the sum over rows of lambda_i plus times c_i minus u_i plus lambda_i minus times l_i minus c_i, plus rho over 2 times the sum of squared violations v_i; a dashed primal step updates the primal proxy. Dual phase, primal proxy frozen: p enters the dual proxy, which outputs lambda-hat; a frozen copy gives lambda^k; with x-hat frozen, the target t (B4) is the projection Pi_M of lambda^k plus rho sigma of x-hat and lambda^k, and the dual loss L_d (B5) is the smoothed norm of lambda-hat minus t on the inequality rows plus that on the equality rows; a dashed dual step updates the dual proxy. Outer iteration k (Algorithm 1): primal phase, violation v_k (8), copy lambda^k, dual phase, then rho is replaced by the minimum of alpha rho and rho max if v_k exceeds tau v_k minus 1 (B7); then k increases by one.">
  </picture>
</p>
<p align="center"><i>The primal proxy minimizes the augmented Lagrangian for the frozen dual proxy; the dual proxy learns the multiplier step computed from a frozen copy of itself; the outer iteration alternates the two phases and raises the penalty when the violation stalls.</i></p>

**Self-supervised primal-dual learning with an augmented Lagrangian.** L2OALM learns a *primal proxy* `x̂(p)` and a *dual proxy* `λ̂(p)` for a family of constrained optimization problems `min f(x; p) s.t. l ≤ c(x; p) ≤ u` without solving any instance: the primal proxy minimizes the augmented Lagrangian for the multipliers predicted by the dual proxy, and the dual proxy learns the augmented-Lagrangian multiplier update, with the penalty increased whenever the violation stalls. This is PDL, Park & Van Hentenryck, [*Self-Supervised Primal-Dual Learning for Constrained Optimization*](https://arxiv.org/abs/2208.09046), AAAI 2023. Problems can be stated in JuMP or in ExaModels; with ExaModels the batch is evaluated by GPU-capable kernels.

## Installation

L2OALM and its dependency `L2OCore` are not registered yet. `L2OCore` is the
`lib/L2OCore` subpackage of LearningToOptimize.jl; the registered
LearningToOptimize 1.0.0 predates it. Install `L2OCore` first, then L2OALM,
on any Julia version from 1.10:

```julia
using Pkg
Pkg.add(url = "https://github.com/LearningToOptimize/LearningToOptimize.jl", subdir = "lib/L2OCore")
Pkg.add(url = "https://github.com/LearningToOptimize/L2OALM.jl")
```

The example below also needs JuMP, Lux, Optimisers and Zygote:
`Pkg.add(["JuMP", "Lux", "Optimisers", "Zygote"])`. The ExaModels path
additionally needs ExaModels and
[BatchNLPKernels.jl](https://github.com/LearningToOptimize/BatchNLPKernels.jl)
(`Pkg.add(url = "https://github.com/LearningToOptimize/BatchNLPKernels.jl")`).

## Example

```julia
using JuMP, L2OCore, L2OALM, Lux, Optimisers, Zygote, Random

model = Model()
@variable(model, p[1:2] in MOI.Parameter(1.0))
@variable(model, -2 <= x[1:3] <= 2)
@constraint(model, x[1] + x[2] + x[3] == p[1])
@constraint(model, x[1] - x[2] <= p[2])
@objective(model, Min, sum(x .^ 2) + x[1] - 3x[3])
problem = ParametricProblem(model)

method = ALMMethod(; ρ_init = 1.0, α = 2.0, τ = 0.8, ρmax = 1e3)
pipe = L2OPipeline(problem, method)
sampler = BoxSampler(fill(-1.0, 2), fill(1.0, 2))
data = generate_data(pipe; sampler, n = 256)          # parameters only

rng = Random.Xoshiro(0)
primal = Chain(Dense(2, 32, tanh), Dense(32, 3))
dual = Chain(Dense(2, 32, tanh), Dense(32, num_duals(problem); init_weight = zeros64, init_bias = zeros64))
(ps, st), (pd, sd) = Lux.setup(rng, primal), Lux.setup(rng, dual)
trainer = ALMTrainer(Training.TrainState(primal, f64(ps), st, Adam(1e-2)),
                     Training.TrainState(dual, f64(pd), sd, Adam(1e-2)))
L2OALM.train!(trainer, pipe, data; outer_iterations = 10, primal_epochs = 15,
              dual_epochs = 15, batchsize = 64, rng)

P = L2OCore.sample_parameters(sampler, rng, 32)
pred = complete(pipe, P, L2OCore.predict_batch(trainer, P))
evaluate(pipe, P, pred)            # objectives and violations per instance
```

## What you need to provide

  - A parametric problem: a JuMP model with `MOI.Parameter` variables
    (LP, convex QP or smooth NLP with element-wise constraint sets), or an
    ExaModel with `parameter`s wrapped in a BatchNLPKernels `BatchModel`.
  - A sampler of parameters (e.g. `BoxSampler`) or a parameter matrix.
  - Two proxies: Lux models for the built-in training loop (any callable for
    custom loops), the dual one with `num_duals(problem)` outputs.

## What you get

  - Losses of the paper as plain array functions (`primal_loss`,
    `dual_target`, `dual_loss`, `violation_measure`, `update_penalty`) that run
    on GPU arrays and differentiate with array AD.
  - The training loop of Algorithm 1 (`L2OALM.train!`, Lux extension; not
    exported, so it does not clash with `Flux.train!`).
  - Multipliers projected onto the correct sign, and variable bounds
    enforced by construction with `SigmoidBoundMap` or `HardSigmoidBoundMap`.
    Feasibility of the other constraints and optimality are measured with
    `evaluate` (violations, objective gaps).

## The LearningToOptimize ecosystem

| Package | What it provides |
|:--|:--|
| [LearningToOptimize.jl](https://github.com/LearningToOptimize/LearningToOptimize.jl) | the penalty method, Flux training and solver-labeled dataset generation; hosts the two subpackages below |
| ↳ [L2OCore](https://github.com/LearningToOptimize/LearningToOptimize.jl/tree/main/lib/L2OCore) | the shared interface: method families, problem wrapper, batched evaluators, datasets |
| ↳ [L2OSobolev](https://github.com/LearningToOptimize/LearningToOptimize.jl/tree/main/lib/L2OSobolev) | derivative (Sobolev) supervision added to any method |
| [L2ODLL.jl](https://github.com/LearningToOptimize/L2ODLL.jl) | Dual Lagrangian Learning: dual proxies that give valid dual bounds for conic problems |
| [L2ODC3.jl](https://github.com/LearningToOptimize/L2ODC3.jl) | DC3: equality completion and inequality correction for primal proxies |
| [L2OALM.jl](https://github.com/LearningToOptimize/L2OALM.jl) | primal-dual learning with an augmented Lagrangian |

L2OALM implements `L2OCore.AbstractPrimalDualMethod` and evaluates ExaModels
problems with [BatchNLPKernels.jl](https://github.com/LearningToOptimize/BatchNLPKernels.jl).

## Citation

If you use L2OALM, please cite the method:

```bibtex
@inproceedings{park2023pdl,
  title     = {Self-Supervised Primal-Dual Learning for Constrained Optimization},
  author    = {Park, Seonho and Van Hentenryck, Pascal},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {37},
  pages     = {4052--4060},
  year      = {2023}
}
```

and the software (see [`CITATION.cff`](CITATION.cff)). Contributions are welcome:
see [`CONTRIBUTING.md`](CONTRIBUTING.md).
