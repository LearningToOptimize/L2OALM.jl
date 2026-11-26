_first_leaf(x::Optimisers.Leaf) = x
function _first_leaf(x::Union{Tuple,NamedTuple})
    for v in values(x)
        l = _first_leaf(v)
        l === nothing || return l
    end
    return nothing
end
_first_leaf(::Any) = nothing

@testset "equality_scale: per-row penalties" begin
    @test_throws ArgumentError ALMMethod(; equality_scale = 0.0)
    rng = Random.Xoshiro(40)
    pipe1 = qp_pipe()
    form = ALMFormulation(pipe1)
    @test row_penalty(pipe1.method, form, 2.0) == fill(2.0, 9)
    pipe3 = qp_pipe(; equality_scale = 3.0)
    @test row_penalty(pipe3.method, form, 2.0) == [6.0; fill(2.0, 8)]
    # DualTargetMatchesEq6 with ρ_h = s ρ on the equality row.
    X = 3 .* randn(rng, 3, 4)
    μ, λ = abs.(randn(rng, 8, 4)), randn(rng, 1, 4)
    g, h = qp_gh(X, P_TEST)
    T = dual_target(pipe3.method, form, P_TEST, X, qp_paper_to_rows(μ, λ); ρ = 2.0)
    @test T ≈ qp_paper_to_rows(max.(μ .+ 2.0 .* g, 0), λ .+ 6.0 .* h)
    # PDLMatchesEq5 with ρ_h = s ρ.
    eq5 =
        qp_obj(X) .+ vec(sum(μ .* g; dims = 1) .+ λ .* h) .+
        vec(1.0 .* sum(max.(g, 0) .^ 2; dims = 1) .+ 3.0 .* h .^ 2)
    @test primal_loss(pipe3, P_TEST, X, qp_paper_to_rows(μ, λ); ρ = 2.0) ≈ sum(eq5) / 4
    # KKTFixedPoint and KKTStationarity hold for any positive per-row penalty.
    for aug in AUGS
        pipe = qp_pipe(; equality_scale = 5.0, augmentation = aug)
        Xs, Λs, _ = kkt_pair(pipe, P_TEST)
        f = ALMFormulation(pipe)
        @test maximum(abs, dual_target(pipe.method, f, P_TEST, Xs, Λs; ρ = 3.0) .- Λs) <
              1e-6
        @test violation_measure(pipe.method, f, P_TEST, Xs, Λs; ρ = 3.0) < 1e-6
        gx =
            Zygote.gradient(x -> primal_loss(pipe.method, f, P_TEST, x, Λs; ρ = 3.0), Xs)[1]
        @test maximum(abs, gx) < 1e-6
    end
end

@testset "instance_violations and violation_report" begin
    rng = Random.Xoshiro(41)
    pipe = qp_pipe()
    form = ALMFormulation(pipe)
    X, Λ = 3 .* randn(rng, 3, 5), randn(rng, 9, 5)
    P = P_TEST[:, rand(rng, 1:4, 5)]
    v = instance_violations(pipe, P, X, Λ; ρ = 2.0)
    @test length(v) == 5 && maximum(v) ≈ violation_measure(pipe, P, X, Λ; ρ = 2.0)
    # ViolationReportMatchesEvaluate
    rep = violation_report(form, P, X)
    ref = evaluate(pipe, P, Prediction(; primal = X))
    @test rep.objective ≈ ref.objective
    @test rep.max_equality_violation ≈ ref.max_equality_violation
    @test rep.max_inequality_violation ≈ ref.max_inequality_violation
    @test rep.max_violation ≈ ref.max_violation
    # works for the kernel evaluator too (no JuMP problem)
    eform = ALMFormulation(qp_exa(8))
    erep = violation_report(eform, P, X)
    @test erep.max_violation ≈ ref.max_violation
    @test erep.objective ≈ ref.objective
end

@testset "violation_reduction and learning-rate decay (App. C.3)" begin
    rng = Random.Xoshiro(42)
    method = ALMMethod(; α = 2.0)
    pipe = L2OPipeline(ParametricProblem(qp_model()), method)
    form = ALMFormulation(pipe)
    P = 2 .* rand(rng, 2, 64) .- 1
    Pval = 2 .* rand(rng, 2, 16) .- 1
    # mean reduction
    tr = lux_trainer(rng, 2, 3, 9, 1.0; width = 8)
    dual0 = deepcopy(tr.dual)
    L2OALM.train!(
        tr,
        pipe,
        P;
        outer_iterations = 1,
        primal_epochs = 2,
        dual_epochs = 2,
        batchsize = 64,
        rng = rng,
        violation_reduction = v -> sum(v) / length(v),
    )
    X = map_primal(
        method,
        form,
        first(tr.primal.model(P, tr.primal.parameters, Lux.testmode(tr.primal.states))),
    )
    Λ0 = first(dual0.model(P, dual0.parameters, Lux.testmode(dual0.states)))
    v = instance_violations(method, form, P, X, Λ0; ρ = 1.0)
    @test tr.history[1].violation ≈ sum(v) / length(v)
    # decay off by default: no options recorded
    @test !haskey(tr.options, :primal_eta)
    @test_throws ArgumentError L2OALM.train!(tr, pipe, P; lr_decay = 0.0)
    # A large learning rate makes the validation loss go up at some epochs;
    # every such epoch multiplies the rate by lr_decay.
    tr = lux_trainer(rng, 2, 3, 9, 1.0; width = 8, lr = 0.5)
    L2OALM.train!(
        tr,
        pipe,
        P;
        outer_iterations = 2,
        primal_epochs = 8,
        dual_epochs = 8,
        batchsize = 16,
        rng = rng,
        validation = Pval,
        lr_decay = 0.5,
    )
    np, nd = get(tr.options, :primal_decays, 0), get(tr.options, :dual_decays, 0)
    @test np + nd > 0
    np > 0 && @test tr.options[:primal_eta] ≈ 0.5 * 0.5^np
    nd > 0 && @test tr.options[:dual_eta] ≈ 0.5 * 0.5^nd
    # the optimiser state really uses the decayed rate
    leaf = _first_leaf(tr.primal.optimizer_state)
    np > 0 && @test leaf.rule.eta ≈ tr.options[:primal_eta]
end
