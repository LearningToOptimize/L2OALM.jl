# # ExaModels and GPU kernels
#
# Nonlinear problems stated in [ExaModels](https://github.com/exanauts/ExaModels.jl)
# are evaluated in batch by
# [BatchNLPKernels](https://github.com/LearningToOptimize/BatchNLPKernels.jl):
# one kernel launch evaluates the objective (or the constraints) of every
# instance of a batch, on CPU or GPU, and its ChainRules rules make the
# evaluation differentiable. L2OALM wraps such a model as an
# [`ALMFormulation`](@ref), so the same losses and training loop apply.
#
# This page runs on the CPU backend. For an NVIDIA GPU, replace
# `KernelAbstractions.CPU()` by `CUDABackend()` (after `using CUDA`), and move
# the parameters and the Lux states to the device with `gpu_device()` (after
# `using LuxCUDA`).

using ExaModels, KernelAbstractions, BatchNLPKernels
using L2OCore, L2OALM
using Lux, Optimisers, Zygote, Random

# ## A parametric nonconvex problem
#
# ``\min_x (x_1 - θ_1)^2 + (x_2 - θ_2)^2 + e^{x_3/2}`` subject to
# ``\|x\|^2 \le 4``, ``x_1 x_2 + \sin x_3 = 1/2`` and ``-3 \le x \le 3``.
# (BatchNLPKernels currently needs one augmented constraint, written with
# `constraint!`.)

batchsize = 32
core = ExaCore(; backend = KernelAbstractions.CPU())
x = variable(core, 3; lvar = -3.0, uvar = 3.0, start = 0.5)
θ = parameter(core, [0.5, 0.5])
constraint(core, x[1]^2 + x[2]^2 + x[3]^2 for i = 1:1; lcon = -Inf, ucon = 4.0)
c2 = constraint(core, x[1] * x[2] - 0.5 for i = 1:1)
constraint!(core, c2, i => sin(x[3]) for i = 1:1)
objective(core, (x[1] - θ[1])^2 + (x[2] - θ[2])^2 + exp(x[3] / 2) for i = 1:1)
bm = BatchModel(
    ExaModel(core; prod = true),
    batchsize;
    config = BatchModelConfig(:viol_grad),
);

# The formulation: rows are the two constraints followed by the three
# variables (bound rows).

form = ALMFormulation(bm)
(form.lower, form.upper)

# ## Losses run through the kernels

method = ALMMethod(; α = 2.0, ρmax = 1e3)
rng = Random.Xoshiro(1)
P = rand(rng, 2, batchsize)
X = randn(rng, 3, batchsize)
Λ = zeros(num_rows(form), batchsize)
primal_loss(method, form, P, X, Λ; ρ = 1.0)

# and are differentiable:

g = Zygote.gradient(X -> primal_loss(method, form, P, X, Λ; ρ = 1.0), X)[1]
size(g)

# ## Training

primal = Chain(Dense(2, 16, tanh), Dense(16, 3))
dual = Chain(
    Dense(2, 16, tanh),
    Dense(16, num_rows(form); init_weight = zeros64, init_bias = zeros64),
)
ps, st = Lux.setup(rng, primal)
pd, sd = Lux.setup(rng, dual)
trainer = ALMTrainer(
    Training.TrainState(primal, f64(ps), st, Optimisers.Adam(1e-2)),
    Training.TrainState(dual, f64(pd), sd, Optimisers.Adam(1e-2)),
)
Ptrain = rand(rng, 2, 128)
L2OALM.train!(
    trainer,
    method,
    form,
    Ptrain;
    outer_iterations = 5,
    primal_epochs = 10,
    dual_epochs = 10,
    batchsize = batchsize,
    rng = rng,
)
[(h.iteration, h.ρ, round(h.violation; sigdigits = 3)) for h in trainer.history]
