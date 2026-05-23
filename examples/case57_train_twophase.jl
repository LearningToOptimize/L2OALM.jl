# examples/case57_train_twophase.jl
#
# Two-phase PDL training for case57 (v26a):
#
#   Phase 1 — skip the degenerate phase:
#     ρ_init = ρ_max = 1e4 (pinned, no growth).  At ρ=1e4 the zero-generation
#     degenerate min costs 2.88M vs. feasible ~40K — mechanically impossible.
#     Run K1 outer iterations.  After this phase the network is in a non-degenerate
#     basin but equalities oscillate at ~2.5 p.u. (reproduces v21a behaviour).
#
#   Phase 2 — drive equalities to convergence:
#     Start from Phase-1 weights (trainer unchanged).  Swap to a new method with
#     ρ_max=1e6, ρ_eq_scale=10, MAX_DUAL=1e4.
#     - ρ grows from trainer.ρ (=1e4 at end of Phase 1) up to 1e6.
#     - ρ_eq_scale=10 makes effective equality ρ up to 1e7, but MAX_DUAL=1e4
#       caps the gradient at lr × 1e4 = 1.0/step — stable throughout.
#     - Phase-1 weights serve as a warm start in the non-degenerate basin.
#     Run K2 outer iterations.
#
# The key difference from the user's original reset idea: we do NOT reset ρ back
# to a small value (which would misalign the dual network).  Instead, Phase 2
# CONTINUES growing ρ from where Phase 1 left it, but with higher ρ_max and
# ρ_eq_scale so equalities get additional convergence pressure.

const K1              = parse(Int,     get(ENV, "PDL_K1",             "40"))
const K2              = parse(Int,     get(ENV, "PDL_K2",             "60"))
const L_PRIMAL        = parse(Int,     get(ENV, "PDL_L_PRIMAL",       "2500"))
const L_DUAL          = parse(Int,     get(ENV, "PDL_L_DUAL",         "2500"))
const WARMUP_EPOCHS   = parse(Int,     get(ENV, "PDL_WARMUP_EPOCHS",  "25000"))
const LR_PRIMAL       = parse(Float64, get(ENV, "PDL_LR_PRIMAL",      "1.0e-4"))
const LR_DUAL         = parse(Float64, get(ENV, "PDL_LR_DUAL",        "1.0e-3"))
const LR_DECAY        = parse(Float64, get(ENV, "PDL_LR_DECAY",       "0.99"))
const BATCH_SIZE      = parse(Int,     get(ENV, "PDL_BATCH_SIZE",     "200"))
const DATASET_SIZE    = parse(Int,     get(ENV, "PDL_DATASET_SIZE",   "5000"))
const TAU             = parse(Float64, get(ENV, "PDL_TAU",             "0.8"))
const ALPHA           = parse(Float64, get(ENV, "PDL_ALPHA",           "2.0"))
const HIDDEN_MULT     = parse(Float64, get(ENV, "PDL_HIDDEN_MULT",     "4.0"))

# Phase 1 hyperparams
const RHO_P1          = parse(Float64, get(ENV, "PDL_RHO_P1",         "10000.0"))
const MAX_DUAL_P1     = parse(Float64, get(ENV, "PDL_MAX_DUAL_P1",    "10000.0"))
const RHO_EQ_SCALE_P1 = parse(Float64, get(ENV, "PDL_RHO_EQ_SCALE_P1","1.0"))

# Phase 2 hyperparams
const RHO_MAX_P2       = parse(Float64, get(ENV, "PDL_RHO_MAX_P2",    "1000000.0"))
const MAX_DUAL_P2      = parse(Float64, get(ENV, "PDL_MAX_DUAL_P2",   "10000.0"))
const RHO_EQ_SCALE_P2  = parse(Float64, get(ENV, "PDL_RHO_EQ_SCALE_P2","10.0"))

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
    Random.seed!(rng, 2026_05_22)

    if !CUDA.functional()
        @warn "CUDA not functional — falling back to CPU."
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

    is_equal  = model.meta.lcon .== model.meta.ucon
    num_equal = count(is_equal)
    @assert all(is_equal[ncon-num_equal+1:ncon])
    @assert !any(is_equal[1:ncon-num_equal])
    @info "Problem dimensions" nbus ngen narc nvar ncon nθ num_equal

    Random.seed!(rng, 0xCA5E5701)
    Θ_train = (T_F(1.0) .+ T_F(0.1) .* randn(rng, T_F, nθ, DATASET_SIZE)) |> dev
    Random.seed!(rng, 0xCA5E5702)
    Θ_test  = (T_F(1.0) .+ T_F(0.1) .* randn(rng, T_F, nθ, DATASET_SIZE)) |> dev

    h = round(Int, HIDDEN_MULT * nvar)
    lvar = T_F.(model.meta.lvar)
    uvar = T_F.(model.meta.uvar)
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

    dual_model = feed_forward_builder(nθ, ncon, [h, h]) |> dev
    ps_dual, st_dual = Lux.setup(rng, dual_model)
    ps_dual = ps_dual |> dev
    st_dual = st_dual |> dev

    train_state_primal = Training.TrainState(primal_model, ps_primal, st_primal, Optimisers.Adam(LR_PRIMAL))
    train_state_dual   = Training.TrainState(dual_model,   ps_dual,   st_dual,   Optimisers.Adam(LR_DUAL))

    data = DataLoader((Θ_train); batchsize = BATCH_SIZE, shuffle = true) .|> dev

    # ---- shared logger ----
    log_dir = joinpath(@__DIR__, "..", "outputs")
    mkpath(log_dir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    log_path = joinpath(log_dir, "training_log_case57_twophase_$(timestamp).csv")
    open(log_path, "w") do io
        println(io, "phase,outer_iter,rho,rho_eq_effective,mean_abs_mu,mean_abs_lambda,",
                    "mean_violations,max_violation_ineq,max_violation_eq,",
                    "max_bound_violation,mean_objs,primal_loss,wallclock_s")
    end
    @info "CSV log path" log_path
    t0 = time()

    function make_eval_fn(phase::Int, method_test)
        test_primal_loss = primal_loss(method_test)
        return function eval_and_log(iter::Int, trainer)
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
            max_viol_ineq   = maximum(Vc_ineq)
            max_viol_eq     = maximum(Vc_eq)
            max_bound       = maximum(Vb_test)
            mean_obj        = mean(objs_test)
            wallclock       = time() - t0
            rho_eq_eff      = trainer.ρ * (phase == 1 ? RHO_EQ_SCALE_P1 : RHO_EQ_SCALE_P2)
            primal_val, _, _ = test_primal_loss(primal_model, ps_p, st_p, (Θ_test, trainer))

            open(log_path, "a") do io
                @printf(io, "%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.3f\n",
                    phase, iter, trainer.ρ, rho_eq_eff,
                    mean_abs_mu, mean_abs_lambda,
                    mean(Vc_test), max_viol_ineq, max_viol_eq, max_bound, mean_obj,
                    Float64(primal_val), wallclock)
            end

            @info @sprintf("Phase %d iter %3d", phase, iter) ρ=trainer.ρ ρ_eq=rho_eq_eff max_viol_ineq max_viol_eq mean_obj
            return Float64(mean(Vc_test))
        end
    end

    # ==========================================================================
    # PHASE 1: fixed ρ = ρ_max = RHO_P1  (skip degenerate phase)
    # ==========================================================================
    @info "=== PHASE 1 ===" K1 RHO_P1 MAX_DUAL_P1 RHO_EQ_SCALE_P1

    method_p1 = ALMMethod(; batch_model = bm_train, num_equal = num_equal,
        max_dual = MAX_DUAL_P1, ρmax = RHO_P1, τ = TAU, α = ALPHA,
        ρ_eq_scale = RHO_EQ_SCALE_P1, use_analytical_dual = true)
    method_p1_test = ALMMethod(; batch_model = bm_test, num_equal = num_equal,
        max_dual = MAX_DUAL_P1, ρmax = RHO_P1, τ = TAU, α = ALPHA,
        ρ_eq_scale = RHO_EQ_SCALE_P1, use_analytical_dual = true)

    trainer = ALMTrainer(primal_model, train_state_primal, dual_model, train_state_dual, RHO_P1)

    eval_p1 = make_eval_fn(1, method_p1_test)
    eval_p1(0, trainer)   # baseline

    train!(method_p1, trainer, data;
        K = K1, L_primal = L_PRIMAL, L_dual = L_DUAL,
        warmup_epochs = WARMUP_EPOCHS,
        lr_primal = LR_PRIMAL, lr_dual = LR_DUAL, lr_decay = LR_DECAY,
        eval_fn = eval_p1,
    )

    @info "Phase 1 complete" trainer.ρ trainer.max_violations

    # ==========================================================================
    # PHASE 2: grow ρ → RHO_MAX_P2 with ρ_eq_scale=RHO_EQ_SCALE_P2
    # Trainer state (weights + ρ from Phase 1) is preserved.
    # ==========================================================================
    @info "=== PHASE 2 ===" K2 RHO_MAX_P2 MAX_DUAL_P2 RHO_EQ_SCALE_P2

    method_p2 = ALMMethod(; batch_model = bm_train, num_equal = num_equal,
        max_dual = MAX_DUAL_P2, ρmax = RHO_MAX_P2, τ = TAU, α = ALPHA,
        ρ_eq_scale = RHO_EQ_SCALE_P2, use_analytical_dual = true)
    method_p2_test = ALMMethod(; batch_model = bm_test, num_equal = num_equal,
        max_dual = MAX_DUAL_P2, ρmax = RHO_MAX_P2, τ = TAU, α = ALPHA,
        ρ_eq_scale = RHO_EQ_SCALE_P2, use_analytical_dual = true)

    eval_p2 = make_eval_fn(2, method_p2_test)

    train!(method_p2, trainer, data;
        K = K2, L_primal = L_PRIMAL, L_dual = L_DUAL,
        warmup_epochs = 0,   # no second warmup — primal already in good basin
        lr_primal = LR_PRIMAL, lr_dual = LR_DUAL, lr_decay = LR_DECAY,
        eval_fn = eval_p2,
    )

    # ---- Final eval ----
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
