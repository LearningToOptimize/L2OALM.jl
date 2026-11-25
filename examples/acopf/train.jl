# PDL for parametric AC optimal power flow (PGLib cases), evaluated with
# BatchNLPKernels on CPU or GPU.
#
#   julia --project=examples/acopf examples/acopf/train.jl [case] [outer_iterations]
#
# GPU: add CUDA and LuxCUDA to this environment and set L2OALM_CUDA=1.

using L2OALM, L2OCore
using BatchNLPKernels, ExaModels, KernelAbstractions
using GPUArraysCore: @allowscalar
using Lux, Optimisers, Zygote, Random
using PowerModels, PGLib
PowerModels.silence()

include(joinpath(@__DIR__, "power.jl"))

const USE_CUDA = get(ENV, "L2OALM_CUDA", "0") == "1"
if USE_CUDA
    using CUDA, LuxCUDA
end
backend = USE_CUDA ? CUDA.CUDABackend() : KernelAbstractions.CPU()
dev = USE_CUDA ? gpu_device() : cpu_device()

case = get(ARGS, 1, "pglib_opf_case14_ieee.m")
K = parse(Int, get(ARGS, 2, "10"))
T = Float32
batchsize = 64
rng = Random.Xoshiro(0)

model, nbus, ngen, narc = create_parametric_ac_power_model(case; backend = backend, T = T)
bm = BatchModel(model, batchsize; config = BatchModelConfig(:viol_grad))
form = ALMFormulation(bm)
nθ, nx, nλ = length(model.θ), model.meta.nvar, num_rows(form)
@info "AC-OPF $case" nθ nx nλ

# Load multipliers in [0.8, 1.2] at every bus.
Ptrain = dev(T(0.8) .+ T(0.4) .* rand(rng, T, nθ, 2048))
Ptest = dev(T(0.8) .+ T(0.4) .* rand(rng, T, nθ, batchsize))

method = ALMMethod(;
    ρ_init = 1.0f0,
    α = 2.0f0,
    τ = 0.8f0,
    ρmax = 1.0f4,
    bounds = SigmoidBoundMap(),
    augmentation = PHRAugmentation(),
)
primal = Chain(Dense(nθ, 256, relu), Dense(256, 256, relu), Dense(256, nx))
dual =
    Chain(Dense(nθ, 256, relu), Dense(256, nλ; init_weight = zeros32, init_bias = zeros32))
ps, st = Lux.setup(rng, primal)
pd, sd = Lux.setup(rng, dual)
trainer = ALMTrainer(
    Training.TrainState(primal, dev(ps), dev(st), Optimisers.Adam(1.0f-3)),
    Training.TrainState(dual, dev(pd), dev(sd), Optimisers.Adam(1.0f-3));
    ρ = method.ρ_init,
)

log_iteration(trainer, entry) = @info "outer iteration" entry

train!(
    trainer,
    method,
    form,
    Ptrain;
    outer_iterations = K,
    primal_epochs = 5,
    dual_epochs = 5,
    batchsize = batchsize,
    rng = rng,
    callback = log_iteration,
)

out = L2OCore.predict_batch(trainer, Ptest)
X = map_primal(method, form, out[1:nx, :])
V = L2OCore.row_violation(form.evaluator, X, Ptest)
mean_objective = sum(L2OCore.batch_objective(form.evaluator, X, Ptest)) / batchsize
max_violation = maximum(V)
@info "test set" mean_objective max_violation
