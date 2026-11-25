@testset "evaluator agrees with the JuMP formulation" begin
    bm = qp_exa(8)
    form = ALMFormulation(bm)
    @test num_rows(form) == 6 && L2OCore.num_decision_variables(form) == 3
    @test form.lower == [0, -Inf, -1, -2, -2, -2]
    @test form.upper == [0, 0, Inf, 2, 2, 2]
    @test form.equality == [1, 0, 0, 0, 0, 0]
    @test occursin("BatchModelEvaluator", sprint(show, form.evaluator))
    ev = form.evaluator
    @test L2OCore.equality_rows(ev) == Bool[1, 0, 0, 0, 0, 0]
    @test L2OCore.bound_rows(ev) == Bool[0, 0, 0, 1, 1, 1]
    jpipe = L2OPipeline(ParametricProblem(qp_model()), ALMMethod())
    jform = ALMFormulation(jpipe)
    rng = Random.Xoshiro(12)
    X = 3 .* randn(rng, 3, 4)
    @test L2OCore.batch_objective(ev, X, P_TEST) ≈
          L2OCore.batch_objective(jform.evaluator, X, P_TEST)
    Vj = L2OCore.row_violation(jform.evaluator, X, P_TEST)
    @test L2OCore.row_violation(ev, X, P_TEST) ≈
          vcat(Vj[1:1, :], Vj[3:3, :], Vj[2:2, :], Vj[4:6, :] .+ Vj[7:9, :])
    # Same PDL loss at mapped multipliers (the PHR term is not invariant under
    # merging a variable's two bound rows into one interval row).
    for aug in (PDLAugmentation(),)
        m = ALMMethod(; augmentation = aug)
        Λj = project_multipliers(m, jform, randn(rng, 9, 4))
        Λj[4:6, :] .= 0                      # one active side per variable row
        Λj[7:9, :] .= abs.(Λj[7:9, :])
        Λe = qp_jump_to_exa(Λj)
        @test primal_loss(m, form, P_TEST, X, Λe; ρ = 3.0) ≈
              primal_loss(m, jform, P_TEST, X, Λj; ρ = 3.0)
    end
end

@testset "KKTFixedPoint and KKTStationarity on ExaModels (QP and NLP)" begin
    for (jm, exa, map) in (
        (qp_model(), qp_exa(8), qp_jump_to_exa),
        (nlp_model(), nlp_exa(8), nlp_jump_to_exa),
    )
        prob = ParametricProblem(jm)
        ref = solve_reference(prob, P_TEST)
        X, Λ = ref.primal, map(from_jump_duals(ref.dual))
        form = ALMFormulation(exa)
        @test L2OCore.batch_objective(form.evaluator, X, P_TEST) ≈ ref.objective
        for aug in AUGS
            m = ALMMethod(; augmentation = aug)
            @test maximum(abs, dual_target(m, form, P_TEST, X, Λ; ρ = 10.0) .- Λ) < 1e-6
            @test violation_measure(m, form, P_TEST, X, Λ; ρ = 10.0) < 1e-6
            g = Zygote.gradient(x -> primal_loss(m, form, P_TEST, x, Λ; ρ = 10.0), X)[1]
            @test maximum(abs, g) < 1e-6
        end
    end
end

@testset "PrimalLossGradient through the kernels" begin
    rng = Random.Xoshiro(13)
    form = ALMFormulation(nlp_exa(8))
    X = randn(rng, 3, 4)
    Λ = randn(rng, num_rows(form), 4)
    for aug in AUGS
        m = ALMMethod(; augmentation = aug)
        f = x -> primal_loss(m, form, P_TEST, x, Λ; ρ = 4.0)
        @test Zygote.gradient(f, X)[1] ≈ fd_grad(f, X) rtol = 1e-6 atol = 1e-8
    end
end

@testset "PDL training with the kernel evaluator (NLP)" begin
    rng = Random.Xoshiro(14)
    B = 32
    form = ALMFormulation(nlp_exa(B))
    method = ALMMethod(; α = 2.0, ρmax = 1e3)
    P = rand(rng, 2, 128)
    tr = lux_trainer(rng, 2, 3, num_rows(form), method.ρ_init; width = 16)
    train!(
        tr,
        method,
        form,
        P;
        outer_iterations = 6,
        primal_epochs = 10,
        dual_epochs = 10,
        batchsize = B,
        rng = rng,
    )
    @test tr.history[end].violation < tr.history[1].violation
    out = L2OCore.predict_batch(tr, P[:, 1:B])
    Λ = project_multipliers(method, form, out[4:end, :])
    @test check(
        DualConeFeasible(),
        form,
        P[:, 1:B],
        Prediction(; dual = to_jump_duals(Λ)),
    ).passed
end
