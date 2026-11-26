```@meta
CurrentModule = L2OALM
DocTestSetup = :(using L2OALM, L2OCore)
```

# L2OALM.jl

*Self-supervised primal-dual learning with an augmented Lagrangian.*

L2OALM trains two neural proxies for a family of constrained optimization
problems `min_x f(x; p) s.t. l ≤ c(x; p) ≤ u` indexed by parameters `p`: a
**primal proxy** `x̂(p)` and a **dual proxy** `λ̂(p)`. No solver is needed to
create training data. The proxies imitate the iterations of the augmented
Lagrangian method: the primal proxy minimizes the augmented Lagrangian for
the multipliers predicted by the dual proxy, and the dual proxy learns the
multiplier update. This is the PDL method of Park & Van Hentenryck,
*Self-Supervised Primal-Dual Learning for Constrained Optimization*, AAAI 2023
([arXiv:2208.09046](https://arxiv.org/abs/2208.09046)).

The package is an `L2OCore` primal-dual method, so it shares problems,
pipelines and evaluators with the other packages in the
[LearningToOptimize](https://github.com/LearningToOptimize) ecosystem.

```mermaid
flowchart LR
    P["parameters p"] --> PP["primal proxy x̂(p)"]
    P --> DP["dual proxy λ̂(p)"]
    PP --> PL["primal loss (5): f + λᵀr + ρ/2 ‖v‖²"]
    DP -- frozen --> PL
    PP -- frozen --> DT["dual target (6): λ_k + ρ σ"]
    DP -- "frozen copy λ_k" --> DT
    DT --> DL["dual loss (7)"]
    DP --> DL
    PL --> V["violation measure (8)"]
    V --> R["penalty rule (9): ρ ← min(αρ, ρmax)"]
    R -.-> PL
    R -.-> DT
```

## Where to start

  - [Theory](@ref): the optimization problem, the losses with the paper's
    equation numbers, and the properties of the method.
  - [Package guide](@ref): getting started with JuMP or ExaModels, the API
    concepts and the extension points.
  - [Design](@ref): how the package uses L2OCore, how GPU evaluation works,
    which machine-learning frameworks are supported, and how other
    primal-dual variants fit.
  - Examples: [PDL on a parametric QP](examples/qp.md) and
    [ExaModels and GPU kernels](examples/examodels.md).

## A first look

```jldoctest
julia> method = ALMMethod(; ρ_init = 1.0, α = 2.0, τ = 0.8, ρmax = 1e3);

julia> enforces_bounds(SigmoidBoundMap())   # variable bounds hold by construction
true

julia> update_penalty(method, 1.0, 0.9, 1.0)   # v_k > τ v_{k-1}: ρ grows by α
2.0

julia> update_penalty(method, 1.0, 0.5, 1.0)   # enough progress: ρ is kept
1.0
```
