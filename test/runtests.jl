using L2OALM
using Test

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
        for i in 1:(length(layer_sizes)-1)
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
    
    function test_penalty_training(; filename="pglib_opf_case14_ieee.m", dev_gpu = gpu_device(), 
        backend=CPU(),
        batch_size = 32,
        dataset_size = 3200,
        rng = Random.default_rng(),
        T = Float64,
    )
        model = create_parametric_ac_power_model(filename; backend = backend, T=T)
        bm = BNK.BatchModel(model, batch_size, config=BNK.BatchModelConfig(:full))
        bm_all = BNK.BatchModel(model, dataset_size, config=BNK.BatchModelConfig(:full))
    
        function PenaltyLoss(model, ps, st, Θ)
            X̂ , st_new = model(Θ, ps, st)
    
            obj = BNK.objective!(bm, X̂, Θ)
            Vc, Vb = BNK.all_violations!(bm, X̂, Θ)
    
            return sum(obj) + 1000 * sum(Vc) + 1000 * sum(Vb), st_new, (;obj=sum(obj), Vc=sum(Vc), Vb=sum(Vb))
        end
    
        nvar = model.meta.nvar
        ncon = model.meta.ncon
        nθ = length(model.θ)
    
        Θ_train = randn(T, nθ, dataset_size) |> dev_gpu
    
        lux_model = feed_forward_builder(nθ, nvar, [320, 320])
    
        ps_model, st_model = Lux.setup(rng, lux_model)
        ps_model = ps_model |> dev_gpu
        st_model = st_model |> dev_gpu
    
        X̂ , _ = lux_model(Θ_train, ps_model, st_model)
    
        y = BNK.objective!(bm_all, X̂, Θ_train)
    
        @test length(y) == dataset_size
        Vc, Vb = BNK.all_violations!(bm_all, X̂, Θ_train)
        @test size(Vc) == (ncon, dataset_size)
        @test size(Vb) == (nvar, dataset_size)
    
        lagrangian_prev = sum(y) + 1000 * sum(Vc) + 1000 * sum(Vb)
    
        train_state = Training.TrainState(lux_model, ps_model, st_model, Optimisers.Adam(1e-5))
    
        data = DataLoader((Θ_train); batchsize=batch_size, shuffle=true) .|> dev_gpu
        for (Θ) in data
            _, loss_val, stats, train_state = Training.single_train_step!(
                AutoZygote(),          # AD backend
                PenaltyLoss,
                (Θ),  # data
                train_state
            )
        end
    
        X̂ , st_model = lux_model(Θ_train, ps_model, st_model)
    
        y = BNK.objective!(bm_all, X̂, Θ_train)
        Vc, Vb = BNK.all_violations!(bm_all, X̂, Θ_train)
        lagrangian_new = sum(y) + 1000 * sum(Vc) + 1000 * sum(Vb)
    
        @test lagrangian_new < lagrangian_prev
    end
    
    @testset "Penalty Training" begin
        backend, dev = if haskey(ENV, "BNK_TEST_CUDA")
            CUDABackend(), gpu_device()
        else
            CPU(), cpu_device()
        end
    
        test_penalty_training(; filename="pglib_opf_case14_ieee.m", dev_gpu = dev, backend=backend, T=Float32)
    end
end
