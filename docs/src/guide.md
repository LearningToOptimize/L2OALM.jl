```@meta
CurrentModule = L2OALM
```

# Package guide

## Installation

L2OALM depends on `L2OCore`, which is not registered yet:

```julia
using Pkg
Pkg.add(url = "https://github.com/LearningToOptimize/LearningToOptimize.jl",
        subdir = "lib/L2OCore", rev = "ar/l2ocore-design")
Pkg.add(url = "https://github.com/LearningToOptimize/L2OALM.jl")
Pkg.add(["JuMP", "Lux", "Optimisers", "Zygote"])   # front end and training
```

For ExaModels problems and GPU evaluation also add
`Pkg.add(url = "https://github.com/LearningToOptimize/BatchNLPKernels.jl")`,
`ExaModels`, `KernelAbstractions` and, on NVIDIA hardware, `CUDA`.

## Getting started (JuMP)

1. **State the problem** as a JuMP model whose instance data are parameters:

   ```julia
   using JuMP, L2OCore, L2OALM
   model = Model()
   @variable(model, p[1:2] in MOI.Parameter(1.0))
   @variable(model, -2 <= x[1:3] <= 2)
   @constraint(model, x[1] + x[2] + x[3] == p[1])
   @constraint(model, x[1] - x[2] <= p[2])
   @objective(model, Min, sum(x .^ 2) + x[1] - 3x[3])
   problem = ParametricProblem(model)
   ```

2. **Bind the method** to the problem. `L2OPipeline` checks that the problem
   class is supported and computes the [`ALMDecomposition`](@ref) (row and
   variable bounds) once:

   ```julia
   method = ALMMethod(; ρ_init = 1.0, α = 2.0, τ = 0.8, ρmax = 1e3)
   pipe = L2OPipeline(problem, method)
   ```

3. **Sample parameters.** PDL is self-supervised, so `generate_data` only
   samples (no solver is called):

   ```julia
   data = generate_data(pipe; sampler = BoxSampler(fill(-1.0, 2), fill(1.0, 2)), n = 1024)
   ```

4. **Create the proxies and train** (Lux extension). The dual proxy outputs
   one multiplier per row (`num_duals(problem)`); start its last layer at zero:

   ```julia
   using Lux, Optimisers, Zygote, Random
   rng = Random.Xoshiro(0)
   primal = Chain(Dense(2, 64, tanh), Dense(64, 3))
   dual = Chain(Dense(2, 64, tanh), Dense(64, num_duals(problem); init_weight = zeros32, init_bias = zeros32))
   ps, st = Lux.setup(rng, primal); pd, sd = Lux.setup(rng, dual)
   trainer = ALMTrainer(Training.TrainState(primal, f64(ps), st, Adam(1e-3)),
                        Training.TrainState(dual, f64(pd), sd, Adam(1e-3)); ρ = method.ρ_init)
   train!(trainer, pipe, data; outer_iterations = 20, primal_epochs = 10, dual_epochs = 10,
          batchsize = 64, rng = rng)
   ```

5. **Predict, complete and evaluate.** `predict_batch` stacks both proxies;
   `complete` applies the bound map and projects the multipliers (returned in
   the `JuMP.dual` sign convention):

   ```julia
   P = L2OCore.sample_parameters(BoxSampler(fill(-1.0, 2), fill(1.0, 2)), rng, 100)
   pred = complete(pipe, P, L2OCore.predict_batch(trainer, P))
   evaluate(pipe, P, pred)                               # objective, violations
   ```

`trainer.history` holds, per outer iteration, the penalty used, the violation
measure of eq. (8) and the mean losses of the last epoch of each phase.

## API concepts

| concept | type / function | role |
|:--|:--|:--|
| method | [`ALMMethod`](@ref) | hyper-parameters only (`ρ_init`, `ρmax`, `α`, `τ`, `max_dual`, augmentation, bound map) |
| problem as ALM sees it | [`ALMFormulation`](@ref) | evaluator + row bounds `l ≤ c(x; p) ≤ u` + sense |
| losses | [`primal_loss`](@ref), [`dual_target`](@ref), [`dual_loss`](@ref) | eqs. (5)–(7); plain array functions, any AD |
| schedule | [`violation_measure`](@ref), [`update_penalty`](@ref) | eqs. (8)–(9) |
| outputs → predictions | [`map_primal`](@ref), [`project_multipliers`](@ref), `L2OCore.complete` | bound map, dual-cone projection |
| training | [`ALMTrainer`](@ref), [`train!`](@ref) | Algorithm 1 (Lux extension) |

Every loss function has two forms: `f(method, form, ...)` (any evaluator,
including ExaModels kernels without JuMP) and `f(pipe, ...)` (JuMP pipelines).
`L2OCore.loss(pipe, inputs, output; ρ, multipliers)` is the primal-phase loss
of a stacked output `[x; λ]`, for code written against the generic L2OCore
interface.

## ExaModels front end and GPU

With `BatchNLPKernels` loaded, `ALMFormulation(bm)` turns a `BatchModel` (an
ExaModel with batch buffers) into a formulation whose objective and
constraints are evaluated by KernelAbstractions kernels and differentiated by
the BatchNLPKernels rules:

```julia
using ExaModels, KernelAbstractions, BatchNLPKernels
core = ExaCore(; backend = CPU())          # CUDABackend() on a GPU
# ... variable, parameter, constraint, objective ...
bm = BatchModel(ExaModel(core; prod = true), batchsize; config = BatchModelConfig(:viol_grad))
form = ALMFormulation(bm)
train!(trainer, method, form, P; batchsize = batchsize)   # batchsize ≤ bm.batch_size
```

On a GPU, build the ExaModel on `CUDABackend()`, move the parameters and the
Lux states to the device with `gpu_device()` (load `LuxCUDA`), and call the
same functions: the losses only use broadcasts and reductions, and the
formulation's bound vectors live on the device. See
[ExaModels and GPU kernels](examples/examodels.md).

Current BatchNLPKernels limitation: the ExaModel must contain at least one
augmented constraint (`constraint!`), otherwise `constraints!` rejects its
empty work buffer.

## Using another ML framework or a custom loop

The losses are plain functions of arrays, so any framework and AD backend can
drive them. A minimal loop with a hand-written proxy and Zygote:

```julia
using Zygote
W = zeros(3, 2); Wd = zeros(num_duals(problem), 2)
form = ALMFormulation(pipe)
Λ = Wd * P                       # frozen multipliers of a (linear) dual proxy
for step in 1:100
    g = Zygote.gradient(W -> primal_loss(method, form, P, W * P, Λ; ρ = 1.0), W)[1]
    W .-= 1e-2 .* g
end
```

[`train!`](@ref) methods for other frameworks are added by dispatching on the
types stored in [`ALMTrainer`](@ref).

## Extension points

  - **A new augmentation** (how multipliers and penalty enter the primal loss):
    subtype [`AbstractAugmentation`](@ref) and add
    `L2OALM.augmentation_term(::MyAug, form, C, Λ, ρ)` returning a `1 × B`
    row. Dual targets, schedule and training are reused.
  - **A new output map** (e.g. a layer that fixes a reference angle): subtype
    [`AbstractBoundMap`](@ref), add [`map_primal`](@ref) and, if the map keeps the
    variables within their bounds, `L2OALM.enforces_bounds(::MyMap) = true`.
  - **A new evaluator** (another kernel backend): subtype
    `L2OCore.AbstractEvaluator`, implement `batch_objective` and
    `constraint_values` (differentiably) and build an
    [`ALMFormulation`](@ref) with the positional constructor.
  - **A new primal-dual method** (another PDL variant, Deep ALM): subtype
    `L2OCore.AbstractPrimalDualMethod`, implement `supports`,
    `decompose`, `complete`, and reuse the formulation and the loss functions
    as needed.
  - **Training for another framework**: add a method of [`train!`](@ref) for
    `ALMTrainer{<:YourState,<:YourState}`.
