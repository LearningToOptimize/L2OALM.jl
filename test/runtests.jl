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
        model, nbus, ngen, blines =
            create_parametric_ac_power_model(filename; backend = backend, T = T)
        bm_train = BNK.BatchModel(model, batch_size, config = BNK.BatchModelConfig(:full))
        bm_test = BNK.BatchModel(model, dataset_size, config = BNK.BatchModelConfig(:full))

        nvar = model.meta.nvar
        ncon = model.meta.ncon
        nθ = length(model.θ)
        num_equal = nbus * 2

        Θ_train = randn(T, nθ, dataset_size) |> dev_gpu
        Θ_test = randn(T, nθ, dataset_size) |> dev_gpu

        primal_model = feed_forward_builder(nθ, nvar, [320, 320])
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

        function validation_testset(
            iter,
            primal_model,
            dual_model,
            train_state_primal,
            train_state_dual,
            hpm_primal,
            hpm_dual;
            max_dual = 1e6,
        )
            ρ = hpm_primal[:ρ]
            X̂_test, _ = primal_model(
                Θ_test,
                train_state_primal.parameters,
                train_state_primal.states,
            )
            objs_test = BNK.objective!(bm_test, X̂_test, Θ_test)
            Vc_test, Vb_test = BNK.all_violations!(bm_test, X̂_test, Θ_test)
            gh_test = BNK.constraints!(bm_test, X̂_test, Θ_test)
            dual_hat, _ =
                dual_model(Θ_test, train_state_dual.parameters, train_state_dual.states)
            # Separate bound and equality constraints
            gh_bound = gh_test[1:end-num_equal, :]
            gh_equal = gh_test[end-num_equal+1:end, :]
            dual_hat_bound = dual_hat[1:end-num_equal, :]
            dual_hat_equal = dual_hat[end-num_equal+1:end, :]

            # Target for dual variables
            dual_target = vcat(
                min.(max.(dual_hat_bound + ρ .* gh_bound, 0), max_dual),
                min.(dual_hat_equal + ρ .* gh_equal, max_dual),
            )

            dual_loss = mean((dual_hat .- dual_target) .^ 2)

            @info "Validation Testset: Iteration $iter" mean(objs_test) mean(Vc_test) mean(
                Vb_test,
            ) dual_loss
            return iter >= 100 ? true : false
        end

        L2OALM_train!(
            bm_train,
            num_equal,
            primal_model,
            dual_model,
            train_state_primal,
            train_state_dual,
            data,
            stopping_criteria = [validation_testset],
        )
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
