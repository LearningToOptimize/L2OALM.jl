"""
    AbstractAugmentation

How the multipliers and the penalty enter the primal loss. Built-in choices:

  - [`PDLAugmentation`](@ref): eq. (5) of Park & Van Hentenryck (2023),
    `λᵀ r(x) + ρ/2 ‖v(x)‖²` with `r` the signed residual and `v` the violation;
  - [`PHRAugmentation`](@ref): the classical Powell-Hestenes-Rockafellar
    augmented Lagrangian.

**Extension point.** Subtype `AbstractAugmentation` and add a method of
[`augmentation_term`](@ref)`(aug, form, C, Λ, ρ)` returning the per-instance
term (a `1 × B` row or length-`B` vector); everything else (dual targets,
penalty schedule, training loop) is shared.
"""
abstract type AbstractAugmentation end

"""
    PDLAugmentation()

Primal loss term of eq. (5) in Park & Van Hentenryck (2023), extended row-wise
to box constraints `l ≤ c ≤ u`:

```math
\\sum_i \\big( λ_i^+ (c_i - u_i) + λ_i^- (l_i - c_i) \\big)
  + \\frac{ρ}{2} \\sum_i \\big( \\max(l_i - c_i, 0) + \\max(c_i - u_i, 0) \\big)^2 .
```

For an inequality `g = c - u ≤ 0` this is `μ g + ρ/2 max(g, 0)²`, for an
equality `h = c - b` it is `λ h + ρ/2 h²`, i.e. exactly eq. (5).
"""
struct PDLAugmentation <: AbstractAugmentation end

"""
    PHRAugmentation()

Powell-Hestenes-Rockafellar augmented Lagrangian term

```math
\\sum_i \\Big( \\frac{ρ}{2} \\operatorname{dist}\\big(c_i + λ_i/ρ, [l_i, u_i]\\big)^2
  - \\frac{λ_i^2}{2ρ} \\Big).
```

Its gradient with respect to `c` is the (unclipped) dual target of eq. (6),
`λ + ρ σ(x, λ)`, so the primal step "sees" the multiplier one ALM update
ahead. This is the classical ALM of Rockafellar (1973) and the form behind
the "analytical dual" variant of PDL discussed for AC-OPF.
"""
struct PHRAugmentation <: AbstractAugmentation end

"""
    AbstractBoundMap

Map applied to the raw primal output of the proxy before evaluation
([`map_primal`](@ref)). Built-in: [`NoBoundMap`](@ref) (identity) and
[`SigmoidBoundMap`](@ref) (variable bounds hold by construction).

**Extension point.** Subtype it, add a method of [`map_primal`](@ref) and, if
the map makes a promise, of `L2OCore.guarantees` for
`ALMMethod{<:Real,<:AbstractAugmentation,YourMap}`.
"""
abstract type AbstractBoundMap end

"""
    NoBoundMap()

Identity: the primal block of the proxy output is the prediction.
"""
struct NoBoundMap <: AbstractBoundMap end

"""
    SigmoidBoundMap()

Maps each raw output `z` into its variable's bounds:
`l + (u - l) σ(z)` for two finite bounds, `l + softplus(z)` for a lower bound
only, `u - softplus(-z)` for an upper bound only, `z` for a free variable
(and the fixed value for a fixed variable), followed by a `clamp` to absorb
rounding. Guarantee: `BoundFeasible` (variable bounds hold exactly). Unlike a
`clamp`, the map has a non-zero gradient everywhere.
"""
struct SigmoidBoundMap <: AbstractBoundMap end

"""
    ALMMethod(; ρ_init = 1.0, ρmax = 1e4, α = 10.0, τ = 0.8, max_dual = 1e6,
              augmentation = PDLAugmentation(), bounds = NoBoundMap(),
              dual_norm_eps = 1e-8)

Self-supervised primal-dual learning (PDL) with an augmented Lagrangian,
Park & Van Hentenryck (AAAI 2023), as an `L2OCore.AbstractPrimalDualMethod`.
A primal proxy `x̂(p)` and a dual proxy `λ̂(p)` are trained alternately
(Algorithm 1 of the paper; see [`train!`](@ref)):

  - primal phase: minimise [`primal_loss`](@ref) (eq. (5)) with `λ̂` frozen;
  - dual phase: regress `λ̂` onto [`dual_target`](@ref) (eq. (6)) with
    [`dual_loss`](@ref) (eq. (7));
  - penalty: [`update_penalty`](@ref) (eq. (9)) from the
    [`violation_measure`](@ref) (eq. (8)).

Hyper-parameters (paper notation):

  - `ρ_init`: initial penalty `ρ`;
  - `ρmax`: safeguard `ρ_max`; `α > 1`: penalty growth factor; `τ ∈ (0, 1)`:
    required violation decrease;
  - `max_dual`: multipliers are clipped to `[-max_dual, max_dual]` (a numerical
    safeguard that is not in the paper);
  - `augmentation`: [`PDLAugmentation`](@ref) (paper) or
    [`PHRAugmentation`](@ref);
  - `bounds`: [`NoBoundMap`](@ref) or [`SigmoidBoundMap`](@ref);
  - `dual_norm_eps`: smoothing of the norms of eq. (7) so the loss is
    differentiable when the prediction equals the target.

Assumptions: the problem class is `LinearConic` (box rows only),
`ConvexQuadratic` or `SmoothNonlinear` (see `L2OCore.supports`); training
needs an evaluator whose `batch_objective` and `constraint_values` are
differentiable (`MatrixEvaluator`, or the BatchNLPKernels evaluator for
nonlinear problems).

Guarantees (see `L2OCore.guarantees`): the completed multipliers always have
the sign of the dual cone of their row ([`DualConeFeasible`](@ref)); with
`bounds = SigmoidBoundMap()` the variable bounds hold (`BoundFeasible`). The
method makes **no** feasibility or optimality promise for the remaining
constraints: PDL drives violations down during training but certifies nothing
on unseen instances.
"""
struct ALMMethod{T<:Real,A<:AbstractAugmentation,B<:AbstractBoundMap} <:
       AbstractPrimalDualMethod
    ρ_init::T
    ρmax::T
    α::T
    τ::T
    max_dual::T
    augmentation::A
    bounds::B
    dual_norm_eps::T
    function ALMMethod(
        ρ_init::T,
        ρmax::T,
        α::T,
        τ::T,
        max_dual::T,
        augmentation::A,
        bounds::B,
        dual_norm_eps::T,
    ) where {T<:Real,A<:AbstractAugmentation,B<:AbstractBoundMap}
        ρ_init > 0 || throw(ArgumentError("ρ_init must be positive"))
        ρmax >= ρ_init || throw(ArgumentError("ρmax must be ≥ ρ_init"))
        α > 1 || throw(ArgumentError("α must be > 1"))
        0 < τ < 1 || throw(ArgumentError("τ must be in (0, 1)"))
        max_dual > 0 || throw(ArgumentError("max_dual must be positive"))
        dual_norm_eps >= 0 || throw(ArgumentError("dual_norm_eps must be ≥ 0"))
        return new{T,A,B}(ρ_init, ρmax, α, τ, max_dual, augmentation, bounds, dual_norm_eps)
    end
end

function ALMMethod(;
    ρ_init::Real = 1.0,
    ρmax::Real = 1e4,
    α::Real = 10.0,
    τ::Real = 0.8,
    max_dual::Real = 1e6,
    augmentation::AbstractAugmentation = PDLAugmentation(),
    bounds::AbstractBoundMap = NoBoundMap(),
    dual_norm_eps::Real = 1e-8,
)
    T = float(promote_type(typeof.((ρ_init, ρmax, α, τ, max_dual, dual_norm_eps))...))
    return ALMMethod(
        T(ρ_init),
        T(ρmax),
        T(α),
        T(τ),
        T(max_dual),
        augmentation,
        bounds,
        T(dual_norm_eps),
    )
end

L2OCore.supports(::ALMMethod, ::L2OCore.LinearConic) = true
L2OCore.supports(::ALMMethod, ::L2OCore.ConvexQuadratic) = true
L2OCore.supports(::ALMMethod, ::L2OCore.SmoothNonlinear) = true

L2OCore.guarantees(::ALMMethod) = (DualConeFeasible(),)
L2OCore.guarantees(::ALMMethod{<:Real,<:AbstractAugmentation,SigmoidBoundMap}) =
    (BoundFeasible(), DualConeFeasible())

L2OCore.decompose(problem::ParametricProblem, ::ALMMethod) = ALMDecomposition(problem)

# ---------------------------------------------------------------------------
# Maps from raw proxy outputs to predictions

_finite_or_zero(x) = ifelse(isfinite(x), x, zero(x))

# softplus(z) = log(1 + exp(z)), overflow-safe
_softplus(z) = max(z, zero(z)) + log1p(exp(-abs(z)))
_sigmoid(z) = inv(one(z) + exp(-z))

function _bounded(z, l, u)
    hl, hu = isfinite(l), isfinite(u)
    lf, uf = _finite_or_zero(l), _finite_or_zero(u)
    y = ifelse(
        hl & hu,
        lf + (uf - lf) * _sigmoid(z),
        ifelse(hl, lf + _softplus(z), ifelse(hu, uf - _softplus(-z), z)),
    )
    return clamp(y, l, u)
end

"""
    map_primal(method::ALMMethod, form::ALMFormulation, Z::AbstractMatrix)
    map_primal(bounds::AbstractBoundMap, form, Z)

Map the raw primal block `Z` (`num_decision_variables × B`) of the proxy output
to a primal prediction, according to `method.bounds`.
"""
map_primal(method::ALMMethod, form::ALMFormulation, Z::AbstractMatrix) =
    map_primal(method.bounds, form, Z)
map_primal(::NoBoundMap, ::ALMFormulation, Z::AbstractMatrix) = Z
map_primal(::SigmoidBoundMap, form::ALMFormulation, Z::AbstractMatrix) =
    _bounded.(Z, form.xlower, form.xupper)

# Lower/upper clip for the multiplier of each row: λᵢ ≤ 0 needs a finite lower
# bound, λᵢ ≥ 0 a finite upper bound (normal cone of [lᵢ, uᵢ]).
_dual_lo(l, M) = ifelse(isfinite(l), -M, zero(M))
_dual_hi(u, M) = ifelse(isfinite(u), M, zero(M))

"""
    project_multipliers(method::ALMMethod, form::ALMFormulation, Λ::AbstractMatrix)

Clip multipliers (L2OALM convention, see [`ALMFormulation`](@ref)) to the sign
pattern of their row and to `[-max_dual, max_dual]`: rows without a finite
upper bound get `λᵢ ≤ 0`, rows without a finite lower bound get `λᵢ ≥ 0`.
This is the projection onto the dual cone, so it is idempotent.
"""
function project_multipliers(method::ALMMethod, form::ALMFormulation, Λ::AbstractMatrix)
    M = convert(eltype(Λ), method.max_dual)
    return clamp.(Λ, _dual_lo.(form.lower, M), _dual_hi.(form.upper, M))
end

"""
    to_jump_duals(Λ)

Convert between the L2OALM multiplier convention (`λ` in the normal cone of the
row's box, see [`ALMFormulation`](@ref)) and the `JuMP.dual` convention used by
`L2OCore.Prediction` (`y = -λ`, for both minimisation and maximisation).
"""
to_jump_duals(Λ) = -Λ

"""
    from_jump_duals(Y)

Inverse of [`to_jump_duals`](@ref): multipliers in the L2OALM convention from
duals in the `JuMP.dual` convention (`λ = -y`).
"""
from_jump_duals(Y) = -Y

# ---------------------------------------------------------------------------
# L2OCore pipeline steps

"""
    L2OCore.complete(method::ALMMethod, pipe, inputs, output) -> Prediction

Split the stacked output `[x; λ]` (`num_decision_variables + num_duals` rows),
apply [`map_primal`](@ref) to the primal block and
[`project_multipliers`](@ref) to the dual block, and return the duals in the
`JuMP.dual` sign convention.
"""
function L2OCore.complete(method::ALMMethod, pipe::L2OPipeline, inputs, output)
    form = ALMFormulation(pipe)
    X, Λ = _split(pipe, output)
    return Prediction(;
        primal = map_primal(method, form, X),
        dual = to_jump_duals(project_multipliers(method, form, Λ)),
    )
end

function _split(pipe::L2OPipeline, output)
    d = pipe.decomposition::ALMDecomposition
    Z = output isa AbstractVector ? reshape(output, :, 1) : output
    size(Z, 1) == L2OCore.output_dimension(d) || throw(
        DimensionMismatch(
            "output has $(size(Z, 1)) rows, expected $(L2OCore.output_dimension(d)) " *
            "($(d.nprimal) primal + $(d.ndual) multipliers)",
        ),
    )
    return Z[1:d.nprimal, :], Z[(d.nprimal+1):end, :]
end

"""
    L2OCore.generate_data(method::ALMMethod, pipe; sampler, n, rng) -> L2ODataset

PDL is self-supervised: the dataset holds only sampled parameters (no solver
is called).
"""
function L2OCore.generate_data(
    ::ALMMethod,
    pipe::L2OPipeline;
    sampler,
    n::Integer,
    rng::Random.AbstractRNG = Random.default_rng(),
    kwargs...,
)
    P = L2OCore.sample_parameters(sampler, rng, n)
    return L2ODataset(P)
end
