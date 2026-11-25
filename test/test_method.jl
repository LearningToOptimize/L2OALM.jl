struct _OtherMethod <: L2OCore.AbstractPrimalMethod end

# A method-agnostic add-on (the L2OCore hook used for derivative supervision).
struct _ConstantTerm{M} <: L2OCore.AbstractMethodWrapper
    inner::M
    c::Float64
end
L2OCore.extra_loss(w::_ConstantTerm, pipe, inputs, output; kwargs...) = w.c
L2OCore.supports(::_OtherMethod, ::L2OCore.ConvexQuadratic) = true

@testset "construction and validation" begin
    m = ALMMethod()
    @test m isa L2OCore.AbstractPrimalDualMethod
    @test L2OCore.method_family(m) == L2OCore.PrimalDualFamily()
    @test m.ρ_init == 1.0 && m.ρmax == 1e4 && m.α == 10.0 && m.τ == 0.8
    @test ALMMethod(;
        ρ_init = 1.0f0,
        ρmax = 10.0f0,
        α = 2.0f0,
        τ = 0.5f0,
        max_dual = 1.0f3,
        dual_norm_eps = 0.0f0,
    ).ρ_init isa Float32
    @test_throws ArgumentError ALMMethod(; ρ_init = 0.0)
    @test_throws ArgumentError ALMMethod(; ρ_init = 10.0, ρmax = 1.0)
    @test_throws ArgumentError ALMMethod(; α = 1.0)
    @test_throws ArgumentError ALMMethod(; τ = 1.0)
    @test_throws ArgumentError ALMMethod(; max_dual = 0.0)
    @test_throws ArgumentError ALMMethod(; dual_norm_eps = -1.0)
end

@testset "supports and guarantees" begin
    m = ALMMethod()
    @test supports(m, LinearConic())
    @test supports(m, ConvexQuadratic())
    @test supports(m, SmoothNonlinear())
    @test !supports(m, MixedInteger())
    @test guarantees(m) == (DualConeFeasible(),)
    @test guarantees(ALMMethod(; bounds = SigmoidBoundMap())) ==
          (BoundFeasible(), DualConeFeasible())
    mi = Model()
    @variable(mi, p in MOI.Parameter(1.0))
    @variable(mi, y, Bin)
    @constraint(mi, y <= p)
    @objective(mi, Min, y)
    @test_throws UnsupportedProblemError L2OPipeline(ParametricProblem(mi), m)
end

@testset "decomposition and formulation" begin
    prob = ParametricProblem(qp_model())
    pipe = L2OPipeline(prob, ALMMethod())
    d = pipe.decomposition
    @test d isa ALMDecomposition
    @test output_dimension(pipe) == 3 + 9
    form = ALMFormulation(pipe)
    @test ALMFormulation(form) === form
    @test num_rows(form) == 9
    @test L2OCore.num_decision_variables(form) == 3
    @test form.lower == [0, -1, -Inf, -2, -2, -2, -Inf, -Inf, -Inf]
    @test form.upper == [0, Inf, 0, Inf, Inf, Inf, 2, 2, 2]
    @test form.equality == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    @test form.xlower == fill(-2, 3) && form.xupper == fill(2, 3)
    @test form.sense == 1
    @test occursin("9 rows", sprint(show, form))
    @test ALMFormulation(
        L2OPipeline(ParametricProblem(qp_model(; sense = MAX_SENSE)), ALMMethod()),
    ).sense == -1
    # Non-ALM pipelines are rejected.
    @test_throws ArgumentError ALMFormulation(L2OPipeline(prob, _OtherMethod()))
    # Dimension checks.
    ev = form.evaluator
    @test_throws DimensionMismatch ALMFormulation(
        ev,
        zeros(2),
        zeros(3),
        zeros(1),
        zeros(1),
        zeros(2),
    )
    @test_throws DimensionMismatch ALMFormulation(
        ev,
        zeros(2),
        zeros(2),
        zeros(1),
        zeros(2),
        zeros(2),
    )
    @test_throws ArgumentError ALMFormulation(
        ev,
        zeros(2),
        zeros(2),
        zeros(1),
        zeros(1),
        zeros(2);
        sense = 2,
    )
end

@testset "fixed variables and unsupported sets" begin
    m = Model()
    @variable(m, p in MOI.Parameter(1.0))
    @variable(m, x[1:2])
    fix(x[2], 0.5)
    @constraint(m, x[1] + x[2] == p)
    @objective(m, Min, x[1]^2)
    form = ALMFormulation(L2OPipeline(ParametricProblem(m), ALMMethod()))
    @test form.xlower == [-Inf, 0.5] && form.xupper == [Inf, 0.5]
    msoc = Model()
    @variable(msoc, p in MOI.Parameter(1.0))
    @variable(msoc, x[1:3])
    @constraint(msoc, [p; x] in SecondOrderCone())
    @objective(msoc, Min, sum(x))
    @test_throws ArgumentError L2OPipeline(ParametricProblem(msoc), ALMMethod())
end

@testset "complete and generate_data" begin
    rng = Random.Xoshiro(1)
    prob = ParametricProblem(qp_model())
    for bounds in (NoBoundMap(), SigmoidBoundMap())
        method = ALMMethod(; bounds = bounds, max_dual = 3.0)
        pipe = L2OPipeline(prob, method)
        P = randn(rng, 2, 7)
        out = 10 .* randn(rng, output_dimension(pipe), 7)
        pred = complete(pipe, P, out)
        @test size(pred.primal) == (3, 7) && size(pred.dual) == (9, 7)
        @test check(DualConeFeasible(), prob, P, pred).passed
        @test maximum(abs, pred.dual) <= 3.0
        for g in guarantees(method)
            @test check(g, prob, P, pred).passed
        end
        bounds isa NoBoundMap && @test pred.primal == out[1:3, :]
        # a single instance as a vector
        @test size(complete(pipe, P[:, 1], out[:, 1]).primal) == (3, 1)
        @test_throws DimensionMismatch complete(pipe, P, out[1:5, :])
    end
    pipe = L2OPipeline(prob, ALMMethod())
    ds = generate_data(
        pipe;
        sampler = BoxSampler(fill(-1.0, 2), fill(1.0, 2)),
        n = 11,
        rng = rng,
    )
    @test ds isa L2ODataset && num_samples(ds) == 11
    @test ds.primal === nothing && ds.dual === nothing
end

@testset "DualConeFeasible detects sign errors" begin
    prob = ParametricProblem(qp_model())
    pipe = L2OPipeline(prob, ALMMethod())
    form = ALMFormulation(pipe)
    P = zeros(2, 1)
    # JuMP convention: GreaterThan rows need y ≥ 0, LessThan rows y ≤ 0.
    y = zeros(9, 1)
    y[2] = -1.0     # GreaterThan with a negative dual
    r = check(DualConeFeasible(), prob, P, Prediction(; dual = y))
    @test !r.passed && r.worst ≈ 1.0
    @test !check(DualConeFeasible(), form, P, Prediction(; dual = y)).passed
    y[2] = 1.0
    y[3] = -2.0
    @test check(DualConeFeasible(), prob, P, Prediction(; dual = y)).passed
    @test check(DualConeFeasible(), form, P, Prediction(; dual = vec(y))).passed
    @test_throws ArgumentError check(
        DualConeFeasible(),
        prob,
        P,
        Prediction(; primal = zeros(3, 1)),
    )
end

@testset "SigmoidBoundMap" begin
    m = Model()
    @variable(m, p in MOI.Parameter(1.0))
    @variable(m, -1 <= a <= 2)
    @variable(m, b >= 3)
    @variable(m, c <= -4)
    @variable(m, d)
    @variable(m, e == 0.25)
    @constraint(m, a + b + c + d + e == p)
    @objective(m, Min, a^2 + b^2 + c^2 + d^2)
    pipe = L2OPipeline(ParametricProblem(m), ALMMethod(; bounds = SigmoidBoundMap()))
    form = ALMFormulation(pipe)
    Z = [-1e3 -1.0 0.0 1.0 1e3]
    Z = repeat(Z, 5, 1)
    X = map_primal(pipe.method, form, Z)
    @test all(-1 .<= X[1, :] .<= 2)
    @test all(X[2, :] .>= 3)
    @test all(X[3, :] .<= -4)
    @test X[4, :] == Z[4, :]
    @test all(X[5, :] .== 0.25)
    @test all(isfinite, X)
    # strictly monotone, non-zero gradient in the interior
    g = Zygote.gradient(
        z -> sum(map_primal(pipe.method, form, z)),
        [0.0; 0.0; 0.0; 0.0; 0.0;;],
    )[1]
    @test all(g[1:4] .> 0)
end

@testset "JuMP dual convention" begin
    Λ = randn(Random.Xoshiro(2), 4, 3)
    @test from_jump_duals(to_jump_duals(Λ)) == Λ
    @test to_jump_duals(Λ) == -Λ
end

@testset "WrapperComposes: L2OCore method wrappers compose with ALMMethod" begin
    prob = ParametricProblem(qp_model())
    inner = ALMMethod(; bounds = SigmoidBoundMap())
    wpipe = L2OPipeline(prob, _ConstantTerm(inner, 0.25))
    pipe = L2OPipeline(prob, inner)
    rng = Random.Xoshiro(30)
    P, out = randn(rng, 2, 3), randn(rng, 12, 3)
    @test guarantees(wpipe.method) == guarantees(inner)
    @test L2OCore.loss(wpipe, P, out; ρ = 2.0, proxy = identity) ≈
          L2OCore.loss(pipe, P, out; ρ = 2.0) + 0.25
    @test complete(wpipe, P, out).dual == complete(pipe, P, out).dual
end
