# Each testset name is a named claim documented in docs/src/theory.md; a formal
# proof can later attach to the same name.

const AUGS = (PDLAugmentation(), PHRAugmentation())
const P_TEST = [0.5 1.0 -0.3 0.8; 0.2 -1.0 0.5 0.0]

qp_pipe(; sense = MIN_SENSE, kw...) =
    L2OPipeline(ParametricProblem(qp_model(; sense = sense)), ALMMethod(; kw...))

function kkt_pair(pipe, P)
    ref = solve_reference(pipe.problem, P)
    @assert all(==(LOCALLY_SOLVED), ref.status)
    return ref.primal, from_jump_duals(ref.dual), ref
end

fd_grad(f, X) = FiniteDifferences.grad(central_fdm(5, 1), f, X)[1]

@testset "KKTFixedPoint: dual update and violation vanish at a KKT pair" begin
    # Assumption: (x*, λ*) is a KKT pair (Ipopt, tol 1e-8). Claim: the ALM
    # multiplier update (eq. 6) is the identity and v (eq. 8) is 0.
    for (name, pipe, P) in (
        ("QP min", qp_pipe(), P_TEST),
        ("QP max", qp_pipe(; sense = MAX_SENSE), P_TEST),
        ("NLP", L2OPipeline(ParametricProblem(nlp_model()), ALMMethod()), P_TEST),
    )
        X, Λ, _ = kkt_pair(pipe, P)
        form = ALMFormulation(pipe)
        for ρ in (0.1, 1.0, 100.0)
            T = dual_target(pipe.method, form, P, X, Λ; ρ = ρ)
            @test maximum(abs, T .- Λ) < 1e-6 * max(1, ρ)
            @test violation_measure(pipe.method, form, P, X, Λ; ρ = ρ) < 1e-6
        end
    end
end

@testset "KKTStationarity: primal loss is stationary at a KKT pair" begin
    # Claim: ∇ₓ L_ρ(x*, λ*) = 0 for both augmentations (eq. 5 and PHR).
    for aug in AUGS, sense in (MIN_SENSE, MAX_SENSE)
        pipe = qp_pipe(; sense = sense, augmentation = aug)
        X, Λ, _ = kkt_pair(pipe, P_TEST)
        form = ALMFormulation(pipe)
        g =
            Zygote.gradient(x -> primal_loss(pipe.method, form, P_TEST, x, Λ; ρ = 10.0), X)[1]
        @test maximum(abs, g) < 1e-6
    end
    # Nonlinear problem with the (non-differentiable) JuMPEvaluator: finite differences.
    for aug in AUGS
        pipe = L2OPipeline(ParametricProblem(nlp_model()), ALMMethod(; augmentation = aug))
        X, Λ, _ = kkt_pair(pipe, P_TEST)
        form = ALMFormulation(pipe)
        g = fd_grad(x -> primal_loss(pipe.method, form, P_TEST, x, Λ; ρ = 10.0), X)
        @test maximum(abs, g) < 1e-5
    end
end

@testset "PrimalLossGradient: Zygote equals finite differences" begin
    rng = Random.Xoshiro(3)
    for aug in AUGS
        pipe = qp_pipe(; augmentation = aug)
        form = ALMFormulation(pipe)
        X = 3 .* randn(rng, 3, 4)            # infeasible points: penalty active
        Λ = randn(rng, num_rows(form), 4)
        f = x -> primal_loss(pipe.method, form, P_TEST, x, Λ; ρ = 5.0)
        @test Zygote.gradient(f, X)[1] ≈ fd_grad(f, X) rtol = 1e-7 atol = 1e-8
        # through the stacked-output L2OCore loss (dual block ignored)
        out = vcat(X, Λ)
        g = Zygote.gradient(o -> L2OCore.loss(pipe, P_TEST, o; ρ = 5.0), out)[1]
        @test g[1:3, :] ≈ Zygote.gradient(f, X)[1]
        @test all(iszero, g[4:end, :])
        @test L2OCore.loss(pipe, P_TEST, out; ρ = 5.0, multipliers = Λ) ≈ f(X)
    end
end

# Literal g ≤ 0 / h = 0 form of qp_model for checking eqs. (5)-(8) of the paper.
function qp_gh(X, P)
    g = vcat(
        .-1 .- (X[2:2, :] .+ 2 .* X[3:3, :]),      # x₂ + 2x₃ ≥ -1
        X[1:1, :] .- X[2:2, :] .- P[2:2, :],       # x₁ - x₂ ≤ p₂
        .-2 .- X,                                  # x ≥ -2
        X .- 2,                                    # x ≤ 2
    )
    h = X[1:1, :] .+ X[2:2, :] .+ X[3:3, :] .- P[1:1, :]
    return g, h
end
qp_obj(X) = vec(sum(X .^ 2; dims = 1) .+ X[1:1, :] .- 3 .* X[3:3, :])
# (μ, λ) of the paper → L2OALM rows [eq, ≥, ≤, lb×3, ub×3].
qp_paper_to_rows(μ, λ) = vcat(λ, .-μ[1:1, :], μ[2:2, :], .-μ[3:5, :], μ[6:8, :])

@testset "PDLMatchesEq5: PDLAugmentation is the paper's primal loss" begin
    rng = Random.Xoshiro(4)
    pipe = qp_pipe()
    form = ALMFormulation(pipe)
    X = 3 .* randn(rng, 3, 4)
    μ, λ = abs.(randn(rng, 8, 4)), randn(rng, 1, 4)
    ρ = 7.0
    g, h = qp_gh(X, P_TEST)
    eq5 =
        qp_obj(X) .+ vec(sum(μ .* g; dims = 1) .+ λ .* h) .+
        ρ / 2 .* vec(sum(max.(g, 0) .^ 2; dims = 1) .+ h .^ 2)
    Λ = qp_paper_to_rows(μ, λ)
    @test primal_loss(pipe.method, form, P_TEST, X, Λ; ρ = ρ) ≈ sum(eq5) / 4
end

@testset "DualTargetMatchesEq6 and ViolationMeasureMatchesEq8" begin
    rng = Random.Xoshiro(5)
    pipe = qp_pipe()
    form = ALMFormulation(pipe)
    X = 3 .* randn(rng, 3, 4)
    μ, λ = abs.(randn(rng, 8, 4)), randn(rng, 1, 4)
    ρ = 3.0
    g, h = qp_gh(X, P_TEST)
    T = dual_target(pipe.method, form, P_TEST, X, qp_paper_to_rows(μ, λ); ρ = ρ)
    @test T ≈ qp_paper_to_rows(max.(μ .+ ρ .* g, 0), λ .+ ρ .* h)
    σ = max.(g, .-μ ./ ρ)
    eq8 = max(maximum(abs, h), maximum(abs, σ))
    @test violation_measure(pipe.method, form, P_TEST, X, qp_paper_to_rows(μ, λ); ρ = ρ) ≈
          eq8
end

@testset "PHRGradientIsDualTarget" begin
    # ∂/∂c of the PHR term is λ + ρσ(x, λ), the unclipped eq. (6) update.
    rng = Random.Xoshiro(6)
    pipe = qp_pipe(; augmentation = PHRAugmentation(), max_dual = 1e12)
    form = ALMFormulation(pipe)
    X = 3 .* randn(rng, 3, 4)
    Λ = project_multipliers(pipe.method, form, randn(rng, num_rows(form), 4))
    C = L2OCore.constraint_values(form.evaluator, X, P_TEST)
    ρ = 2.5
    gC =
        Zygote.gradient(c -> sum(augmentation_term(PHRAugmentation(), form, c, Λ, ρ)), C)[1]
    @test gC ≈ dual_target(pipe.method, form, P_TEST, X, Λ; ρ = ρ)
end

@testset "PenaltyLossAtZeroMultipliers" begin
    # With λ = 0 both augmentations reduce to f + ρ/2 ‖v(x)‖² (eq. (4) of the paper).
    rng = Random.Xoshiro(7)
    X = 3 .* randn(rng, 3, 4)
    for aug in AUGS
        pipe = qp_pipe(; augmentation = aug)
        form = ALMFormulation(pipe)
        V = L2OCore.row_violation(form.evaluator, X, P_TEST)
        ref = sum(qp_obj(X) .+ 2.0 .* vec(sum(V .^ 2; dims = 1))) / 4
        @test primal_loss(pipe.method, form, P_TEST, X, zeros(num_rows(form), 4); ρ = 4.0) ≈
              ref
    end
end

@testset "SenseInvariance: max f equals min -f" begin
    rng = Random.Xoshiro(8)
    X = randn(rng, 3, 4)
    Λ = randn(rng, 9, 4)
    pmin, pmax = qp_pipe(), qp_pipe(; sense = MAX_SENSE)
    @test primal_loss(pmin, P_TEST, X, Λ; ρ = 2.0) ≈
          primal_loss(pmax, P_TEST, X, Λ; ρ = 2.0)
    @test dual_target(pmin, P_TEST, X, Λ; ρ = 2.0) ≈
          dual_target(pmax, P_TEST, X, Λ; ρ = 2.0)
end

@testset "DualTargetInCone: targets satisfy the sign pattern and max_dual" begin
    rng = Random.Xoshiro(9)
    pipe = qp_pipe(; max_dual = 5.0)
    form = ALMFormulation(pipe)
    for _ = 1:20
        X = 4 .* randn(rng, 3, 6)
        Λ = 10 .* randn(rng, 9, 6)
        ρ = exp(4 * randn(rng))
        T = dual_target(pipe, P_TEST[:, rand(rng, 1:4, 6)], X, Λ; ρ = ρ)
        @test check(
            DualConeFeasible(),
            form,
            nothing,
            Prediction(; dual = to_jump_duals(T)),
        ).passed
        @test maximum(abs, T) <= 5.0
        Πλ = project_multipliers(pipe.method, form, Λ)
        @test project_multipliers(pipe.method, form, Πλ) == Πλ     # idempotent
    end
end

@testset "PenaltyMonotone: eq. (9) never decreases ρ and respects ρmax" begin
    rng = Random.Xoshiro(10)
    m = ALMMethod(; ρ_init = 1.0, ρmax = 50.0, α = 3.0, τ = 0.5)
    ρ, vprev = m.ρ_init, Inf
    for _ = 1:200
        v = abs(randn(rng))
        ρnew = update_penalty(m, ρ, v, vprev)
        @test ρ <= ρnew <= max(ρ, m.ρmax)
        @test (ρnew > ρ) == (v > m.τ * vprev && ρ < m.ρmax)
        ρ, vprev = ρnew, v
    end
    @test update_penalty(m, 1.0, 1.0, Inf) == 1.0        # k = 1: no update
    @test update_penalty(m, 1.0, 1.0, 1.0) == 3.0
    @test update_penalty(m, 40.0, 1.0, 1.0) == 50.0
end

@testset "DualLossMatchesEq7" begin
    rng = Random.Xoshiro(11)
    for ε in (0.0, 1e-8)
        pipe = qp_pipe(; dual_norm_eps = ε)
        form = ALMFormulation(pipe)
        Λ, T = randn(rng, 9, 5), randn(rng, 9, 5)
        D = Λ .- T
        eq7 = sum(sqrt.(sum(D[2:end, :] .^ 2; dims = 1)) .+ abs.(D[1:1, :])) / 5
        @test dual_loss(pipe, Λ, T) ≈ eq7 rtol = 1e-6
        @test dual_loss(pipe, T, T) == 0
    end
    # differentiable at the target with smoothing
    pipe = qp_pipe()
    T = randn(rng, 9, 3)
    g = Zygote.gradient(l -> dual_loss(pipe, l, T), T)[1]
    @test all(isfinite, g) && maximum(abs, g) == 0
end

@testset "ValidDualBoundStrictlyConvex: sign-feasible multipliers bound a strictly convex QP" begin
    # Assumption: Q ≻ 0 (here Q = 2I), so the Lagrangian is bounded below in x for
    # every multiplier; then any DualConeFeasible prediction gives a valid bound
    # by weak duality (L2OCore.dual_bound), even an untrained one.
    rng = Random.Xoshiro(15)
    pipe = qp_pipe()
    ref = solve_reference(pipe.problem, P_TEST)
    for _ = 1:5
        out = 5 .* randn(rng, output_dimension(pipe), size(P_TEST, 2))
        pred = complete(pipe, P_TEST, out)
        @test check(
            ValidDualBound(),
            pipe.problem,
            P_TEST,
            pred;
            reference = ref.objective,
        ).passed
    end
end

# ---------------------------------------------------------------------------
# Claims added by the theory audit against the paper's full text.

@testset "BoundRowsNotDualized: Y is handled by the bound map, as in the paper" begin
    # Paper, Sec. 3.2/4 and App. C.2: only g and h get multipliers; the
    # domain Y is enforced by the output layer. With a bound-enforcing map the
    # variable-bound rows get zero multipliers and zero dual targets.
    rng = Random.Xoshiro(16)
    for bounds in (SigmoidBoundMap(), HardSigmoidBoundMap())
        pipe = qp_pipe(; bounds = bounds)
        form = ALMFormulation(pipe)
        @test form.bound == [0, 0, 0, 1, 1, 1, 1, 1, 1]
        @test enforces_bounds(bounds)
        Λ = project_multipliers(pipe.method, form, 10 .* randn(rng, 9, 4))
        @test all(iszero, Λ[4:9, :]) && any(!iszero, Λ[1:3, :])
        X = map_primal(pipe.method, form, 5 .* randn(rng, 3, 4))
        T = dual_target(pipe.method, form, P_TEST, X, 10 .* randn(rng, 9, 4); ρ = 3.0)
        @test all(iszero, T[4:9, :])
        pred = complete(pipe, P_TEST, 5 .* randn(rng, 12, 4))
        @test all(iszero, pred.dual[4:9, :])
        for g in guarantees(pipe.method)
            @test check(g, pipe.problem, P_TEST, pred).passed
        end
    end
    # Without a bound map the bounds are ordinary rows with multipliers.
    @test !enforces_bounds(NoBoundMap())
    pipe = qp_pipe()
    Λ = project_multipliers(pipe.method, ALMFormulation(pipe), ones(9, 1))
    @test Λ[7:9, 1] == ones(3)
end

@testset "PHRMatchesPDLOnEqualityAndViolatedRows" begin
    # The two augmentations coincide on equality rows and on violated
    # inequality rows (docs, Theory: "(B2) vs (B3)").
    pipe = qp_pipe()
    form = ALMFormulation(pipe)
    P = P_TEST[:, 1:1]
    X = [3.0; 3.0; 3.0;;]                  # x > 2: all upper bounds violated
    Λ = zeros(9, 1)
    Λ[1] = 0.7                              # equality row
    Λ[7:9] .= [0.2, 0.5, 1.0]               # violated upper-bound rows
    C = L2OCore.constraint_values(form.evaluator, X, P)
    @test augmentation_term(PDLAugmentation(), form, C, Λ, 4.0) ≈
          augmentation_term(PHRAugmentation(), form, C, Λ, 4.0)
    # and differ on a strictly satisfied inequality with a positive multiplier
    X2 = [0.0; 0.0; 0.0;;]
    C2 = L2OCore.constraint_values(form.evaluator, X2, P)
    @test !(
        augmentation_term(PDLAugmentation(), form, C2, Λ, 4.0) ≈
        augmentation_term(PHRAugmentation(), form, C2, Λ, 4.0)
    )
end

@testset "HardSigmoidBoundMap is the paper's convex combination" begin
    pipe = qp_pipe(; bounds = HardSigmoidBoundMap())
    form = ALMFormulation(pipe)
    Z = [-10.0 -3.0 0.0 1.5 3.0 10.0]
    X = map_primal(pipe.method, form, repeat(Z, 3, 1))
    @test X[1, :] ≈ -2 .+ 4 .* clamp.(vec(Z) ./ 6 .+ 0.5, 0, 1)
    @test all(-2 .<= X .<= 2)
end
