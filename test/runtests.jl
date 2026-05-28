using L2OALM
using Test
using Lux
using Optimisers
using MLUtils
using KernelAbstractions
using ExaModels
using BatchNLPKernels
using CUDA

using PowerModels
PowerModels.silence()
using PGLib

using Random
using Statistics

import GPUArraysCore: @allowscalar

const BNK = BatchNLPKernels

include("power.jl")

@testset "L2OALM.jl" begin
    function feed_forward_builder(
        num_p::Integer,
        num_y::Integer,
        hidden_layers::AbstractVector{<:Integer};
        activation = relu,
    )
        """
        Builds a Chain of Dense layers with Lux
        """
        # Combine all layers: input size, hidden sizes, output size
        layer_sizes = [num_p; hidden_layers; num_y]

        # Build up a list of Dense layers
        dense_layers = Any[]
        for i = 1:(length(layer_sizes)-1)
            if i < length(layer_sizes) - 1
                # Hidden layers with activation
                push!(dense_layers, Dense(layer_sizes[i], layer_sizes[i+1], activation))
            else
                # Final layer with no activation
                push!(dense_layers, Dense(layer_sizes[i], layer_sizes[i+1]))
            end
        end

        return Chain(dense_layers...)
    end

    function test_alm_training(;
        filename = "pglib_opf_case14_ieee.m",
        dev_gpu = gpu_device(),
        backend = CPU(),
        batch_size = 32,
        dataset_size = 3200,
        rng = Random.default_rng(),
        T = Float64,
    )
        model, nbus, ngen, blines, _ =
            create_parametric_ac_power_model(filename; backend = backend, T = T)
        bm_train = BNK.BatchModel(model, batch_size, config = BNK.BatchModelConfig(:full))
        bm_test = BNK.BatchModel(model, dataset_size, config = BNK.BatchModelConfig(:full))

        nvar = model.meta.nvar
        ncon = model.meta.ncon
        nθ = length(model.θ)
        # Compute equality count from model metadata — equality iff lcon == ucon.
        # Constraint ordering in power.jl puts inequalities first, equalities last,
        # so num_equal is the count of equalities at the tail of the constraint vector.
        is_equal = model.meta.lcon .== model.meta.ucon
        num_equal = count(is_equal)
        @test all(is_equal[ncon-num_equal+1:ncon])   # ordering invariant: equalities at the tail
        @test !any(is_equal[1:ncon-num_equal])       # ordering invariant: inequalities at the head
        @info "Constraint counts" ncon num_equal num_inequal=(ncon-num_equal)

        # Paper-style sampling: load multipliers ~ 1.0 ± 10%
        # (The model declares `parameter(c, [1.0 for b in data.bus])` so Θ is a
        #  per-bus load multiplier centered at 1.0.)
        Random.seed!(rng, 0x12345678)
        Θ_train = (T(1.0) .+ T(0.1) .* randn(rng, T, nθ, dataset_size)) |> dev_gpu
        Random.seed!(rng, 0xCAFEBABE)
        Θ_test  = (T(1.0) .+ T(0.1) .* randn(rng, T, nθ, dataset_size)) |> dev_gpu

        # Primal architecture: 2-layer MLP + BoundedOutput head to enforce
        # variable bounds architecturally (Park & Van Hentenryck 2023 AC-OPF model).
        lvar = T.(model.meta.lvar)
        uvar = T.(model.meta.uvar)
        primal_model = Chain(
            feed_forward_builder(nθ, nvar, [320, 320]),
            BoundedOutput(lvar, uvar),
        ) |> dev_gpu
        ps_primal, st_primal = Lux.setup(rng, primal_model)
        ps_primal = ps_primal |> dev_gpu
        st_primal = st_primal |> dev_gpu

        dual_model = feed_forward_builder(nθ, ncon, [320, 320])
        ps_dual, st_dual = Lux.setup(rng, dual_model)
        ps_dual = ps_dual |> dev_gpu
        st_dual = st_dual |> dev_gpu

        X̂, _ = primal_model(Θ_test, ps_primal, st_primal)

        y = BNK.objective!(bm_test, X̂, Θ_test)

        @test length(y) == dataset_size
        Vc, Vb = BNK.all_violations!(bm_test, X̂, Θ_test)
        @test size(Vc) == (ncon, dataset_size)
        @test size(Vb) == (nvar, dataset_size)

        # lagrangian_prev = sum(y) + 1000 * sum(Vc) + 1000 * sum(Vb)

        train_state_primal =
            Training.TrainState(primal_model, ps_primal, st_primal, Optimisers.Adam(1e-5))
        train_state_dual =
            Training.TrainState(dual_model, ps_dual, st_dual, Optimisers.Adam(1e-5))

        data = DataLoader((Θ_train); batchsize = batch_size, shuffle = true) .|> dev_gpu

        # Paper hyperparameters for AC-OPF (Park & Van Hentenryck 2023, Table for AC-OPF):
        # α=2 (not 10), ρ_max=1e4 — prevents the augmented Lagrangian from blowing up.
        method = ALMMethod(; batch_model=bm_train, num_equal=num_equal,
            max_dual=1.0e6, ρmax=1.0e4, τ=0.8, α=2.0)
        trainer = ALMTrainer(primal_model, train_state_primal, dual_model, train_state_dual)
        method_test = ALMMethod(; batch_model=bm_test, num_equal=num_equal,
            max_dual=1.0e6, ρmax=1.0e4, τ=0.8, α=2.0)
        test_primal_loss = primal_loss(method_test)
        test_dual_loss = dual_loss(method_test)

        prev_primal_loss_val, _, prev_stats = test_primal_loss(primal_model, ps_primal, st_primal, (Θ_test, trainer))

        primal_loss_val = prev_primal_loss_val
        dual_loss_val   = zero(prev_primal_loss_val)
        stats_primal    = prev_stats
        # L_primal/L_dual now count gradient steps (not epochs); use length(data) per
        # outer iter to match the old behaviour of one full pass per outer iteration.
        n_steps = length(data)
        for iter in 1:100
            single_train_step!(method, trainer, data; L_primal=n_steps, L_dual=n_steps)
            # Log using the trained parameters stored in trainer
            ps_primal_trained = trainer.primal_training_state.parameters
            st_primal_trained = trainer.primal_training_state.states
            ps_dual_trained   = trainer.dual_training_state.parameters
            st_dual_trained   = trainer.dual_training_state.states
            primal_loss_val, _, stats_primal = test_primal_loss(primal_model, ps_primal_trained, st_primal_trained, (Θ_test, trainer))
            dual_loss_val, _, stats_dual     = test_dual_loss(dual_model, ps_dual_trained, st_dual_trained, (Θ_test, trainer))

            @info "Validation Testset: Iteration $iter" primal_loss_val stats_primal.max_violation stats_primal.max_bound_violation stats_primal.mean_violations stats_primal.mean_objs dual_loss_val
        end
        # Check that training drove down constraint violations.
        # The AL loss itself can grow because the trained dual produces large multipliers
        # against any residual violation; mean_violations is the metric ALM directly targets.
        @test stats_primal.mean_violations < prev_stats.mean_violations

        return
    end

    @testset "Penalty Training" begin
        backend, dev = if haskey(ENV, "BNK_TEST_CUDA")
            CUDABackend(), gpu_device()
        else
            CPU(), cpu_device()
        end

        test_alm_training(;
            filename = "pglib_opf_case14_ieee.m",
            dev_gpu = dev,
            backend = backend,
            T = Float32,
        )
    end
end
