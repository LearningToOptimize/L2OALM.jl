# Small test problems, each in JuMP and (where needed) ExaModels form.

"""
Convex QP with parametric right-hand sides:

    min  ‖x‖² + x₁ - 3x₃   (or max of the negation)
    s.t. x₁ + x₂ + x₃ = p₁,  x₁ - x₂ ≤ p₂,  x₂ + 2x₃ ≥ -1,  -2 ≤ x ≤ 2.

JuMP row order (dual_layout): [eq, ≥, ≤, x ≥ -2 (3), x ≤ 2 (3)].
"""
function qp_model(; sense = MIN_SENSE)
    m = Model()
    @variable(m, p[i = 1:2] in MOI.Parameter(1.0))
    @variable(m, -2 <= x[1:3] <= 2)
    @constraint(m, x[1] + x[2] + x[3] == p[1])
    @constraint(m, x[1] - x[2] <= p[2])
    @constraint(m, x[2] + 2x[3] >= -1)
    f = sum(x .^ 2) + x[1] - 3x[3]
    sense == MIN_SENSE ? @objective(m, Min, f) : @objective(m, Max, -f)
    return m
end

"""
Smooth nonconvex NLP:

    min  (x₁ - p₁)² + (x₂ - p₂)² + exp(x₃ / 2)
    s.t. x₁² + x₂² + x₃² ≤ 4,  x₁ x₂ + sin(x₃) = 1/2,  -3 ≤ x ≤ 3.

JuMP row order: [eq (nonlinear), ≤ (nonlinear), x ≥ -3 (3), x ≤ 3 (3)].
"""
function nlp_model()
    m = Model()
    @variable(m, p[i = 1:2] in MOI.Parameter(0.5))
    @variable(m, -3 <= x[1:3] <= 3, start = 0.5)
    @constraint(m, x[1]^2 + x[2]^2 + x[3]^2 <= 4)
    @constraint(m, x[1] * x[2] + sin(x[3]) == 0.5)
    @objective(m, Min, (x[1] - p[1])^2 + (x[2] - p[2])^2 + exp(x[3] / 2))
    return m
end

function _jump_row_order(problem)
    # (set type, row) for each dual row, used to map JuMP rows to ExaModels rows.
    out = Tuple{Any,Int}[]
    for (c, rows) in zip(L2OCore.constraint_refs(problem), L2OCore.dual_layout(problem))
        S = typeof(JuMP.moi_set(JuMP.constraint_object(c)))
        for r in rows
            push!(out, (S, r))
        end
    end
    return out
end

# BatchNLPKernels needs at least one augmented constraint (`constraint!`) in
# the model (an empty `cons_work` buffer is rejected), so the equality is
# written as a base constraint plus an augmentation.
function qp_exa(batch_size)
    c = ExaCore(; backend = KernelAbstractions.CPU())
    x = variable(c, 3; lvar = -2.0, uvar = 2.0)
    θ = parameter(c, [1.0, 1.0])
    s = constraint(c, x[1] + x[2] - θ[1] for i = 1:1)
    constraint!(c, s, i => x[3] for i = 1:1)
    constraint(c, x[1] - x[2] - θ[2] for i = 1:1; lcon = -Inf, ucon = 0.0)
    constraint(c, x[2] + 2x[3] for i = 1:1; lcon = -1.0, ucon = Inf)
    objective(c, x[i]^2 for i = 1:3)
    objective(c, x[1] - 3x[3] for i = 1:1)
    em = ExaModel(c; prod = true)
    return BatchModel(em, batch_size; config = BatchModelConfig(:viol_grad))
end

# Map JuMP multipliers (L2OALM convention, qp_model row order) to qp_exa rows
# [eq, ≤, ≥, x₁, x₂, x₃] (a variable row carries lower + upper multipliers).
qp_jump_to_exa(Λ) = vcat(Λ[1:1, :], Λ[3:3, :], Λ[2:2, :], Λ[4:6, :] .+ Λ[7:9, :])

function nlp_exa(batch_size)
    c = ExaCore(; backend = KernelAbstractions.CPU())
    x = variable(c, 3; lvar = -3.0, uvar = 3.0, start = 0.5)
    θ = parameter(c, [0.5, 0.5])
    constraint(c, x[1]^2 + x[2]^2 + x[3]^2 for i = 1:1; lcon = -Inf, ucon = 4.0)
    s = constraint(c, x[1] * x[2] - 0.5 for i = 1:1)
    constraint!(c, s, i => sin(x[3]) for i = 1:1)
    objective(c, (x[1] - θ[1])^2 + (x[2] - θ[2])^2 + exp(x[3] / 2) for i = 1:1)
    em = ExaModel(c; prod = true)
    return BatchModel(em, batch_size; config = BatchModelConfig(:viol_grad))
end

# nlp_model rows: [eq, ≤, lb (3), ub (3)] → nlp_exa rows [≤, eq, x (3)].
nlp_jump_to_exa(Λ) = vcat(Λ[2:2, :], Λ[1:1, :], Λ[3:5, :] .+ Λ[6:8, :])

function solve_reference(problem, P)
    return solve_instances(problem, P; optimizer = Ipopt.Optimizer, dual = true)
end
