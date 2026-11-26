# Equation numbers refer to Park & Van Hentenryck, "Self-Supervised Primal-Dual
# Learning for Constrained Optimization", AAAI 2023 (arXiv:2208.09046), with the
# row-wise box form `l ≤ c(x) ≤ u` described in docs/src/theory.md.
#
# All functions take batches with the batch in the last dimension and use only
# broadcasts and reductions, so they run on GPU arrays and differentiate with
# array AD (Zygote, Enzyme, ...).

_mean(v) = sum(v) / length(v)

"""
    augmentation_term(aug::AbstractAugmentation, form, C, Λ, ρ) -> 1 × B

Per-instance multiplier and penalty term of the primal loss for constraint
values `C` (`num_rows × B`) and multipliers `Λ` (same shape, already projected
with [`project_multipliers`](@ref)). See [`PDLAugmentation`](@ref) and
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
    return sum(lin .+ (ρ / 2) .* V .^ 2; dims = 1)
end

function augmentation_term(::PHRAugmentation, form::ALMFormulation, C, Λ, ρ)
    Z = C .+ Λ ./ ρ
    D = max.(form.lower .- Z, 0) .+ max.(Z .- form.upper, 0)   # dist(Z, [l, u])
    return sum((ρ / 2) .* D .^ 2 .- Λ .^ 2 ./ (2ρ); dims = 1)
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
    A = augmentation_term(method.augmentation, form, C, Λp, T(ρ))
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
    return project_multipliers(method, form, Λp .+ T(ρ) .* _sigma(form, C, Λp, T(ρ)))
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
    T = eltype(X)
    Λp = project_multipliers(method, form, Λ)
    C = L2OCore.constraint_values(form.evaluator, X, P)
    S = _sigma(form, C, Λp, T(ρ))
    return isempty(S) ? zero(T) : maximum(abs, S)
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

for f in (:primal_loss, :dual_target, :violation_measure)
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
