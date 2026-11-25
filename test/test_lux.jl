function lux_trainer(rng, nin, nx, nλ, ρ; width = 32, lr = 1e-2)
    primal = Chain(Dense(nin, width, tanh), Dense(width, width, tanh), Dense(width, nx))
    # Dual proxy with a zero last layer: the first primal phase is the penalty method.
    dual = Chain(
        Dense(nin, width, tanh),
        Dense(width, nλ; init_weight = zeros64, init_bias = zeros64),
    )
    ps, st = Lux.setup(rng, primal)
    pd, sd = Lux.setup(rng, dual)
    return ALMTrainer(
        Lux.Training.TrainState(primal, f64(ps), st, Optimisers.Adam(lr)),
        Lux.Training.TrainState(dual, f64(pd), sd, Optimisers.Adam(lr));
        ρ = ρ,
    )
end

@testset "PDL on a parametric QP vs Ipopt" begin
    rng = Random.Xoshiro(42)
    method = ALMMethod(; ρ_init = 1.0, α = 2.0, τ = 0.8, ρmax = 1e3)
    pipe = L2OPipeline(ParametricProblem(qp_model()), method)
    sampler = BoxSampler(fill(-1.0, 2), fill(1.0, 2))
    train = generate_data(pipe; sampler = sampler, n = 256, rng = rng)
    Ptest = L2OCore.sample_parameters(sampler, rng, 32)
    ref = solve_reference(pipe.problem, Ptest)
    tr = lux_trainer(rng, 2, 3, num_duals(pipe.problem), method.ρ_init)
    calls = Ref(0)
    train!(
        tr,
        pipe,
        train;
        outer_iterations = 10,
        primal_epochs = 15,
        dual_epochs = 15,
        batchsize = 64,
        rng = rng,
        callback = (t, e) -> (calls[] += 1),
    )
    @test tr.iteration == 10 && length(tr.history) == 10 && calls[] == 10
    hist = tr.history
    @test all(diff([h.ρ for h in hist]) .>= 0)                  # PenaltyMonotone
    @test hist[end].violation < hist[1].violation / 4
    pred = complete(pipe, Ptest, L2OCore.predict_batch(tr, Ptest))
    @test check(DualConeFeasible(), pipe.problem, Ptest, pred).passed
    rep = evaluate(pipe, Ptest, pred)
    gap = abs.(rep.objective .- ref.objective) ./ max.(1, abs.(ref.objective))
    @test sum(gap) / length(gap) < 0.05
    @test sum(rep.max_violation) / length(gap) < 0.05
    # train_proxy! entry point and continuation
    L2OCore.train_proxy!(
        method,
        pipe,
        tr,
        train.inputs;
        outer_iterations = 1,
        batchsize = 64,
        rng = rng,
    )
    @test tr.iteration == 11
end

@testset "SigmoidBoundMap keeps bounds during training" begin
    rng = Random.Xoshiro(7)
    method = ALMMethod(; bounds = SigmoidBoundMap())
    pipe = L2OPipeline(ParametricProblem(qp_model()), method)
    P = 2 .* rand(rng, 2, 64) .- 1
    tr = lux_trainer(rng, 2, 3, num_duals(pipe.problem), 1.0; width = 8)
    train!(
        tr,
        pipe,
        P;
        outer_iterations = 2,
        primal_epochs = 2,
        dual_epochs = 2,
        batchsize = 32,
        rng = rng,
    )
    pred = complete(pipe, P, L2OCore.predict_batch(tr, P))
    for g in guarantees(method)
        @test check(g, pipe.problem, P, pred).passed
    end
end
