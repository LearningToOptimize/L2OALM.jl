# Equation numbers refer to Park & Van Hentenryck, "Self-Supervised Primal-Dual
# Learning for Constrained Optimization", AAAI 2023 (arXiv:2208.09046), with the
# row-wise box form `l ≤ c(x) ≤ u` described in docs/src/theory.md.
#
# All functions take batches with the batch in the last dimension and use only
# broadcasts and reductions, so they run on GPU arrays and differentiate with
# array AD (Zygote, Enzyme, ...).

_mean(v) = sum(v) / length(v)

"""
    row_penalty(method::ALMMethod, form::ALMFormulation, ρ) -> num_rows vector

Penalty of each row: `ρ` for inequality and bound rows, `s ρ` for equality rows
with `s = method.equality_scale`. With the default `s = 1` every row uses the
paper's single `ρ`. A positive per-row penalty keeps the method's properties: KKT points stay
fixed points of the multiplier update and stationary points of the losses,
row by row (see the Theory page).
"""
row_penalty(method::ALMMethod, form::ALMFormulation, ρ::Real) =
    ρ .* (1 .+ (method.equality_scale - 1) .* form.equality)

"""
    augmentation_term(aug::AbstractAugmentation, form, C, Λ, ρ) -> 1 × B

Per-instance multiplier and penalty term of the primal loss for constraint
values `C` (`num_rows × B`), multipliers `Λ` (same shape, already projected
with [`project_multipliers`](@ref)) and penalty `ρ`, a scalar or a per-row
vector ([`row_penalty`](@ref)); implementations must broadcast over `ρ`. See [`PDLAugmentation`](@ref) and
[`PHRAugmentation`](@ref).
"""
function augmentation_term(::PDLAugmentation, form::ALMFormulation, C, Λ, ρ)
    l, u = form.lower, form.upper
    # λ⁺ (c - u) + λ⁻ (l - c): the λᵀg + λᵀh terms of eq. (5). Infinite bounds
    # are replaced by 0; their multiplier side is 0 after projection.
    lin =
        max.(Λ, 0) .* (C .- _finite_or_zero.(u)) .+
        max.(.-Λ, 0) .* (_finite_or_zero.(l) .- C)
    # ν(g) = max(g, 0)², ν(h) = h² of eq. (5), row-wise.
    V = max.(l .- C, 0) .+ max.(C .- u, 0)
    return sum(lin .+ (ρ ./ 2) .* V .^ 2; dims = 1)
end

function augmentation_term(::PHRAugmentation, form::ALMFormulation, C, Λ, ρ)
    Z = C .+ Λ ./ ρ
    D = max.(form.lower .- Z, 0) .+ max.(Z .- form.upper, 0)   # dist(Z, [l, u])
    return sum((ρ ./ 2) .* D .^ 2 .- Λ .^ 2 ./ (2 .* ρ); dims = 1)
end

"""
    primal_loss(method::ALMMethod, form::ALMFormulation, P, X, Λ; ρ) -> Real
    primal_loss(pipe::L2OPipeline, P, X, Λ; ρ)

Primal loss of eq. (5), averaged over the batch:

```math
\\frac{1}{B} \\sum_{k=1}^{B} \\Big( s f(x_k; p_k) + A_ρ\\big(c(x_k; p_k), λ_k\\big) \\Big),
```

where `A_ρ` is [`augmentation_term`](@ref) for `method.augmentation`, `P` the
parameters (`num_parameters × B`), `X` the primal points (already mapped with
[`map_primal`](@ref)) and `Λ` the multipliers of the frozen dual proxy (L2OALM
convention). `Λ` is projected with [`project_multipliers`](@ref) first.
Differentiable in `X` when the evaluator is.
"""
function primal_loss(method::ALMMethod, form::ALMFormulation, P, X, Λ; ρ::Real)
    ev = form.evaluator
    T = eltype(X)
    Λp = project_multipliers(method, form, Λ)
    C = L2OCore.constraint_values(ev, X, P)
    f = form.sense .* L2OCore.batch_objective(ev, X, P)
    A = augmentation_term(method.augmentation, form, C, Λp, row_penalty(method, form, T(ρ)))
    return _mean(vec(A) .+ f)
end

# σ(x, λ) = c - Π_[l,u](c + λ/ρ): for g = c - u it is max(g, -μ/ρ) (σ of eq. (8)),
# for an equality it is h. The ALM multiplier update is λ + ρ σ.
_sigma(form::ALMFormulation, C, Λ, ρ) = C .- clamp.(C .+ Λ ./ ρ, form.lower, form.upper)

"""
    dual_target(method::ALMMethod, form::ALMFormulation, P, X, Λ; ρ) -> num_rows × B
    dual_target(pipe::L2OPipeline, P, X, Λ; ρ)

ALM multiplier update of eq. (6) for primal points `X` and the previous
multipliers `Λ` (the frozen copy of the dual proxy), in box form:

```math
λ^{k+1} = Π_M\\big(λ^k + ρ \\, σ(x, λ^k)\\big), \\qquad
σ(x, λ) = c(x) - Π_{[l,u]}\\big(c(x) + λ/ρ\\big),
```

which is `max(μ + ρ g, 0)` for an inequality `g ≤ 0` and `λ + ρ h` for an
equality `h = 0`. `Π_M` is [`project_multipliers`](@ref) (sign pattern and
`max_dual`).
"""
function dual_target(method::ALMMethod, form::ALMFormulation, P, X, Λ; ρ::Real)
    T = eltype(X)
    Λp = project_multipliers(method, form, Λ)
    C = L2OCore.constraint_values(form.evaluator, X, P)
    R = row_penalty(method, form, T(ρ))
    return project_multipliers(method, form, Λp .+ R .* _sigma(form, C, Λp, R))
end

"""
    dual_loss(method::ALMMethod, form::ALMFormulation, Λ, target) -> Real
    dual_loss(pipe::L2OPipeline, Λ, target)

Dual loss of eq. (7), averaged over the batch: the norm of the inequality part
plus the norm of the equality part of `Λ - target`,

```math
\\frac{1}{B} \\sum_k \\Big( \\|(λ_k - t_k)_{\\mathcal G}\\|_ε + \\|(λ_k - t_k)_{\\mathcal H}\\|_ε \\Big),
\\qquad \\|v\\|_ε = \\sqrt{\\|v\\|_2^2 + ε^2} - ε,
```

with `ε = method.dual_norm_eps` (smoothing, not in the paper, so that the loss
is differentiable when `λ_k = t_k`).
"""
function dual_loss(method::ALMMethod, form::ALMFormulation, Λ, target)
    T = eltype(Λ)
    ε = T(method.dual_norm_eps)
    D2 = (Λ .- target) .^ 2
    w = form.equality
    s_eq = sum(w .* D2; dims = 1)
    s_in = sum((1 .- w) .* D2; dims = 1)
    return _mean(sqrt.(s_eq .+ ε^2) .+ sqrt.(s_in .+ ε^2) .- 2ε)
end

"""
    violation_measure(method::ALMMethod, form::ALMFormulation, P, X, Λ; ρ) -> Real
    violation_measure(pipe::L2OPipeline, P, X, Λ; ρ)

Violation measure of eq. (8) on a batch: `max_k ‖σ(x_k, λ_k)‖_∞` with `σ` as
in [`dual_target`](@ref), i.e. `max(‖h‖_∞, ‖max(g, -μ/ρ)‖_∞)`. The paper takes
the maximum over the whole training set; take the maximum of the batch values.
It is zero exactly when `x` is feasible and complementary with `λ`.
"""
function violation_measure(method::ALMMethod, form::ALMFormulation, P, X, Λ; ρ::Real)
    v = instance_violations(method, form, P, X, Λ; ρ = ρ)
    return isempty(v) ? zero(eltype(X)) : maximum(v)
end

"""
    instance_violations(method::ALMMethod, form::ALMFormulation, P, X, Λ; ρ) -> B vector

Per-instance violation `‖σ(x_k, λ_k)‖_∞` of eq. (8) (before the maximum over
the training set). [`violation_measure`](@ref) is its maximum; `train!`'s
`violation_reduction` keyword chooses another reduction (e.g. the mean).
"""
function instance_violations(method::ALMMethod, form::ALMFormulation, P, X, Λ; ρ::Real)
    T = eltype(X)
    Λp = project_multipliers(method, form, Λ)
    C = L2OCore.constraint_values(form.evaluator, X, P)
    S = _sigma(form, C, Λp, row_penalty(method, form, T(ρ)))
    return size(S, 1) == 0 ? fill!(similar(S, size(S, 2)), 0) :
           vec(maximum(abs.(S); dims = 1))
end

"""
    violation_report(form::ALMFormulation, P, X) -> NamedTuple

Per-instance quality of primal points `X` for any formulation (also those
without a JuMP problem, e.g. ExaModels): `objective` (in the model's sense),
`max_equality_violation`, `max_inequality_violation` (inequality and bound
rows) and `max_violation`, each a length-`B` vector on the formulation's
device. For JuMP pipelines it agrees with `L2OCore.evaluate` (test
`ViolationReportMatchesEvaluate`).
"""
function violation_report(form::ALMFormulation, P, X)
    ev = form.evaluator
    C = L2OCore.constraint_values(ev, X, P)
    V = max.(form.lower .- C, 0) .+ max.(C .- form.upper, 0)
    _colmax(M) =
        size(M, 1) == 0 ? fill!(similar(M, size(M, 2)), 0) : vec(maximum(M; dims = 1))
    eqv = _colmax(form.equality .* V)
    ineqv = _colmax((1 .- form.equality) .* V)
    return (
        objective = L2OCore.batch_objective(ev, X, P),
        max_equality_violation = eqv,
        max_inequality_violation = ineqv,
        max_violation = max.(eqv, ineqv),
    )
end

"""
    update_penalty(method::ALMMethod, ρ, v, v_prev) -> ρ_new

Penalty rule of eq. (9): `min(α ρ, ρmax)` if `v > τ v_prev`, else `ρ`.
Use `v_prev = Inf` at the first outer iteration (the rule applies for `k > 1`).

The penalty never decreases and never exceeds the safeguard: `ρ ≤ ρ_new ≤ max(ρ, ρmax)`.
"""
function update_penalty(method::ALMMethod, ρ::Real, v::Real, v_prev::Real)
    return v > method.τ * v_prev ? max(ρ, min(method.α * ρ, method.ρmax)) : ρ
end

for f in (:primal_loss, :dual_target, :violation_measure, :instance_violations)
    @eval $f(pipe::L2OPipeline, P, X, Λ; ρ::Real) =
        $f(pipe.method, ALMFormulation(pipe), P, X, Λ; ρ = ρ)
end
dual_loss(pipe::L2OPipeline, Λ, target) =
    dual_loss(pipe.method, ALMFormulation(pipe), Λ, target)

"""
    L2OCore.loss(method::ALMMethod, pipe, inputs, output; ρ = method.ρ_init,
                 multipliers = nothing, kwargs...)

The primal-phase loss (eq. (5)) of a stacked raw output `[x; λ]`: the primal
block goes through [`map_primal`](@ref); the multipliers are `multipliers`
(L2OALM convention) when given, else the dual block of `output` with its
derivative ignored, since the dual proxy is frozen in the primal phase. The
dual phase uses [`dual_loss`](@ref) and [`dual_target`](@ref); the full
alternating scheme is [`train!`](@ref).
"""
function L2OCore.loss(
    method::ALMMethod,
    pipe::L2OPipeline,
    inputs,
    output;
    ρ::Real = method.ρ_init,
    multipliers = nothing,
    kwargs...,
)
    form = ALMFormulation(pipe)
    Zx, Zλ = _split(pipe, output)
    Λ = multipliers === nothing ? ChainRulesCore.ignore_derivatives(Zλ) : multipliers
    return primal_loss(method, form, inputs, map_primal(method, form, Zx), Λ; ρ = ρ)
end
