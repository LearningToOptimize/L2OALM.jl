# examples/case57_train.jl
#
# Trains the PDL primal-dual networks on `pglib_opf_case57_ieee` on GPU using
# the paper's AC-OPF hyperparameters and the L2OALM `train!` schedule
# (penalty-only warm-start, K outer ALM iterations × L inner gradient epochs
# per phase, validation-loss-triggered lr decay).
#
# Acceptance: `max_violation` on a held-out test set < 0.01 after training.
#
# Usage:
#   julia --project=. examples/case57_train.jl
#
# Tunables — overridable via env vars (PDL_K, PDL_L_PRIMAL, etc.) for quick smoke runs.
# L_PRIMAL / L_DUAL / WARMUP_EPOCHS are GRADIENT STEP counts, not epoch counts.
# With BATCH_SIZE=200 and DATASET_SIZE=5000 there are 25 batches/epoch.
# Paper (Park & Van Hentenryck 2023): K=10, L=5000, ρ_init=10, α=2, τ=0.8.
# We use K=20, L=2500 as a halfway point: same total steps as K=10/L=5000 but
# with twice as many multiplier updates.  Key fix vs earlier runs: BoundedOutput
# now uses sigmoid (not hardsigmoid) so generators can always recover from
# saturation — hardsigmoid's zero-gradient dead zone at z<-3 caused collapse.
const K               = parse(Int,     get(ENV, "PDL_K",             "20"))
const L_PRIMAL        = parse(Int,     get(ENV, "PDL_L_PRIMAL",      "2500"))
const L_DUAL          = parse(Int,     get(ENV, "PDL_L_DUAL",        "2500"))
const WARMUP_EPOCHS   = parse(Int,     get(ENV, "PDL_WARMUP_EPOCHS", "2500"))
const LR_PRIMAL       = parse(Float64, get(ENV, "PDL_LR_PRIMAL",     "1.0e-4"))
const LR_DUAL         = parse(Float64, get(ENV, "PDL_LR_DUAL",       "1.0e-4"))
const LR_DECAY        = parse(Float64, get(ENV, "PDL_LR_DECAY",      "0.99"))
const BATCH_SIZE      = parse(Int,     get(ENV, "PDL_BATCH_SIZE",    "200"))
const DATASET_SIZE    = parse(Int,     get(ENV, "PDL_DATASET_SIZE",  "5000"))
const RHO_INIT        = parse(Float64, get(ENV, "PDL_RHO_INIT",      "10.0"))
const RHO_MAX         = parse(Float64, get(ENV, "PDL_RHO_MAX",        "1.0e4"))
const TAU             = parse(Float64, get(ENV, "PDL_TAU",             "0.8"))
const ALPHA           = parse(Float64, get(ENV, "PDL_ALPHA",           "2.0"))
const RHO_EQ_SCALE    = parse(Float64, get(ENV, "PDL_RHO_EQ_SCALE",   "1.0"))
const MAX_DUAL        = parse(Float64, get(ENV, "PDL_MAX_DUAL",        "1.0e6"))
const HIDDEN_MULT          = parse(Float64, get(ENV, "PDL_HIDDEN_MULT",          "1.2"))
const USE_ANALYTICAL_DUAL  = parse(Bool,    get(ENV, "PDL_USE_ANALYTICAL_DUAL",  "false"))
const USE_DUAL_LEARNING    = parse(Bool,    get(ENV, "PDL_USE_DUAL_LEARNING",    "true"))
const T_F               = Float32

using L2OALM
using Lux
using Optimisers
using MLUtils
using KernelAbstractions
using ExaModels
using BatchNLPKernels
using CUDA
using NNlib

using PowerModels
PowerModels.silence()
using PGLib

using Random
using Statistics
using Dates
using Printf

import GPUArraysCore: @allowscalar
const BNK = BatchNLPKernels

# AC-OPF model construction (constraint ordering enforced: inequalities first,
# equalities last). Reuses test/power.jl.
include(joinpath(@__DIR__, "..", "test", "power.jl"))

function feed_forward_builder(num_p, num_y, hidden_layers; activation = relu)
    layer_sizes = [num_p; hidden_layers; num_y]
    layers = Any[]
    for i in 1:length(layer_sizes)-1
        if i < length(layer_sizes) - 1
            push!(layers, Dense(layer_sizes[i], layer_sizes[i+1], activation))
        else
            push!(layers, Dense(layer_sizes[i], layer_sizes[i+1]))
        end
    end
    return Chain(layers...)
end

function main()
    rng = Random.default_rng()
    Random.seed!(rng, 2026_05_14)

    if !CUDA.functional()
        @warn "CUDA not functional — falling back to CPU. Performance will be poor."
    end

    backend, dev = CUDA.functional() ?
        (CUDABackend(), gpu_device()) :
        (CPU(),         cpu_device())

    @info "Building parametric AC-OPF model" case = "pglib_opf_case57_ieee.m" T = T_F backend
    model, nbus, ngen, narc, ref_bus_idxs = create_parametric_ac_power_model(
        "pglib_opf_case57_ieee.m"; backend = backend, T = T_F,
    )
    bm_train = BNK.BatchModel(model, BATCH_SIZE,   config = BNK.BatchModelConfig(:full))
    bm_test  = BNK.BatchModel(model, DATASET_SIZE, config = BNK.BatchModelConfig(:full))

    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ   = length(model.θ)

    is_equal = model.meta.lcon .== model.meta.ucon
    num_equal = count(is_equal)
    @assert all(is_equal[ncon-num_equal+1:ncon])  "constraint ordering: equalities at tail"
    @assert !any(is_equal[1:ncon-num_equal])      "constraint ordering: inequalities at head"
    @info "Problem dimensions" nbus ngen narc nvar ncon nθ num_equal

    # Θ: load multipliers ~ 1.0 ± 10%
    Random.seed!(rng, 0xCA5E5701)
    Θ_train = (T_F(1.0) .+ T_F(0.1) .* randn(rng, T_F, nθ, DATASET_SIZE)) |> dev
    Random.seed!(rng, 0xCA5E5702)
    Θ_test  = (T_F(1.0) .+ T_F(0.1) .* randn(rng, T_F, nθ, DATASET_SIZE)) |> dev

    # Primal network with bound-enforcing head; paper width = 1.2·nvar.
    h = round(Int, HIDDEN_MULT * nvar)
    lvar = T_F.(model.meta.lvar)
    uvar = T_F.(model.meta.uvar)
    # Build FixRefBus mask on the same device as lvar (GPU when CUDA is available).
    # FixRefBus.mask must be a CuArray — Lux |> dev only transfers parameters/states,
    # not data fields of custom AbstractLuxLayer structs.
    ref_mask = fill!(similar(lvar), one(T_F))
    @allowscalar for idx in ref_bus_idxs
        ref_mask[idx] = zero(T_F)
    end
    primal_model = Chain(
        feed_forward_builder(nθ, nvar, [h, h]),
        BoundedOutput(lvar, uvar),
        FixRefBus(ref_mask),
    ) |> dev
    ps_primal, st_primal = Lux.setup(rng, primal_model)
    ps_primal = ps_primal |> dev
    st_primal = st_primal |> dev

    # Dual network: 2-layer MLP outputting ncon multipliers.
    dual_model = feed_forward_builder(nθ, ncon, [h, h]) |> dev
    ps_dual, st_dual = Lux.setup(rng, dual_model)
    ps_dual = ps_dual |> dev
    st_dual = st_dual |> dev

    train_state_primal = Training.TrainState(primal_model, ps_primal, st_primal, Optimisers.Adam(LR_PRIMAL))
    train_state_dual   = Training.TrainState(dual_model,   ps_dual,   st_dual,   Optimisers.Adam(LR_DUAL))

    data = DataLoader((Θ_train); batchsize = BATCH_SIZE, shuffle = true) .|> dev

    # Paper hyperparameters for AC-OPF.
    method = ALMMethod(; batch_model = bm_train, num_equal = num_equal,
        max_dual = MAX_DUAL, ρmax = RHO_MAX, τ = TAU, α = ALPHA, ρ_eq_scale = RHO_EQ_SCALE,
        use_analytical_dual = USE_ANALYTICAL_DUAL, use_dual_learning = USE_DUAL_LEARNING)
    trainer = ALMTrainer(primal_model, train_state_primal, dual_model, train_state_dual, RHO_INIT)

    method_test = ALMMethod(; batch_model = bm_test, num_equal = num_equal,
        max_dual = MAX_DUAL, ρmax = RHO_MAX, τ = TAU, α = ALPHA, ρ_eq_scale = RHO_EQ_SCALE,
        use_analytical_dual = USE_ANALYTICAL_DUAL, use_dual_learning = USE_DUAL_LEARNING)
    test_primal_loss = primal_loss(method_test)

    # ---- CSV logger setup ----
    log_dir = joinpath(@__DIR__, "..", "outputs")
    mkpath(log_dir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    log_path = joinpath(log_dir, "training_log_case57_$(timestamp).csv")
    open(log_path, "w") do io
        println(io, "outer_iter,rho,mean_abs_mu,mean_abs_lambda,",
                    "mean_violations,max_violation_ineq,max_violation_eq,",
                    "max_bound_violation,mean_objs,primal_loss,wallclock_s")
    end
    @info "CSV log path" log_path

    t0 = time()
    function eval_and_log(iter::Int, trainer)
        ps_p = trainer.primal_training_state.parameters
        st_p = trainer.primal_training_state.states
        X̂_test, _ = primal_model(Θ_test, ps_p, st_p)
        Vc_test, Vb_test = BNK.all_violations!(bm_test, X̂_test, Θ_test)
        objs_test = BNK.objective!(bm_test, X̂_test, Θ_test)
        dual_hat, _ = trainer.dual_model(Θ_test,
            trainer.dual_training_state.parameters,
            trainer.dual_training_state.states)
        μ = dual_hat[1:end-num_equal, :]
        λ = dual_hat[end-num_equal+1:end, :]
        Vc_ineq = Vc_test[1:end-num_equal, :]
        Vc_eq   = Vc_test[end-num_equal+1:end, :]
        mean_abs_mu     = mean(abs.(μ))
        mean_abs_lambda = mean(abs.(λ))
        mean_viol       = mean(Vc_test)
        max_viol_ineq   = maximum(Vc_ineq)
        max_viol_eq     = maximum(Vc_eq)
        max_bound       = maximum(Vb_test)
        mean_obj        = mean(objs_test)
        wallclock       = time() - t0
        primal_val, _, _ = test_primal_loss(primal_model, ps_p, st_p, (Θ_test, trainer))

        open(log_path, "a") do io
            @printf(io, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.3f\n",
                iter, trainer.ρ, mean_abs_mu, mean_abs_lambda,
                mean_viol, max_viol_ineq, max_viol_eq, max_bound, mean_obj,
                Float64(primal_val), wallclock)
        end

        @info @sprintf("iter %3d", iter) ρ=trainer.ρ max_viol_ineq max_viol_eq mean_viol mean_obj mean_abs_mu mean_abs_lambda
        return Float64(mean_viol)   # signal for lr-decay: decay only when violations worsen
    end

    eval_and_log(0, trainer)   # baseline before training

    @info "Starting train!" K L_PRIMAL L_DUAL WARMUP_EPOCHS RHO_INIT RHO_MAX TAU ALPHA RHO_EQ_SCALE MAX_DUAL LR_PRIMAL LR_DUAL HIDDEN_MULT h USE_ANALYTICAL_DUAL USE_DUAL_LEARNING
    train!(method, trainer, data;
        K = K, L_primal = L_PRIMAL, L_dual = L_DUAL,
        warmup_epochs = WARMUP_EPOCHS,
        lr_primal = LR_PRIMAL, lr_dual = LR_DUAL,
        lr_decay  = LR_DECAY,
        eval_fn   = eval_and_log,
    )

    # ---- Acceptance check ----
    ps_p = trainer.primal_training_state.parameters
    st_p = trainer.primal_training_state.states
    X̂_final, _ = primal_model(Θ_test, ps_p, st_p)
    Vc_final, Vb_final = BNK.all_violations!(bm_test, X̂_final, Θ_test)
    max_viol = maximum(Vc_final)
    max_bound = maximum(Vb_final)
    @info "FINAL" max_viol max_bound log = log_path
    @info "Acceptance" passed = (max_viol < 0.01) target = 0.01

    return max_viol
end

main()
