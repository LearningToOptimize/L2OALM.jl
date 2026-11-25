"""
    DualConeFeasible()

Every multiplier of the dual prediction has the sign required by its row's
set, in the `JuMP.dual` convention: `y ≥ 0` for `GreaterThan`/`Nonnegatives`
rows, `y ≤ 0` for `LessThan`/`Nonpositives` rows, any sign for equalities and
two-sided intervals. [`ALMMethod`](@ref) ensures it by construction
([`project_multipliers`](@ref) in `L2OCore.complete`).

This is **not** `L2OCore.ValidDualBound`: PDL multipliers satisfy the sign
conditions but not Lagrangian stationarity, so they do not certify a bound.

`check(DualConeFeasible(), problem_or_form, inputs, prediction; atol)` returns
per instance the largest sign violation.
"""
struct DualConeFeasible <: AbstractGuarantee end

function _cone_error(lower, upper, Λ)
    # Λ in the L2OALM convention; violation of λᵢ ≤ 0 (no upper bound) and
    # λᵢ ≥ 0 (no lower bound).
    E =
        ifelse.(isfinite.(upper), zero(eltype(Λ)), max.(Λ, 0)) .+
        ifelse.(isfinite.(lower), zero(eltype(Λ)), max.(.-Λ, 0))
    return size(E, 1) == 0 ? zeros(eltype(E), size(E, 2)) : vec(maximum(E; dims = 1))
end

function _check_result(g, errors, atol)
    e = collect(float.(errors))
    worst = isempty(e) ? zero(eltype(e)) : maximum(e)
    return CheckResult(g, worst <= atol, worst, e)
end

_dualbatch(prediction) =
    prediction.dual === nothing ?
    throw(ArgumentError("DualConeFeasible needs a dual prediction")) :
    (prediction.dual isa AbstractVector ? reshape(prediction.dual, :, 1) : prediction.dual)

function L2OCore.check(
    g::DualConeFeasible,
    problem::ParametricProblem,
    inputs,
    prediction;
    atol::Real = 1e-6,
    kwargs...,
)
    d = ALMDecomposition(problem)
    Λ = from_jump_duals(_dualbatch(prediction))
    return _check_result(g, _cone_error(d.lower, d.upper, Λ), atol)
end

function L2OCore.check(
    g::DualConeFeasible,
    form::ALMFormulation,
    inputs,
    prediction;
    atol::Real = 1e-6,
    kwargs...,
)
    Λ = from_jump_duals(_dualbatch(prediction))
    return _check_result(g, collect(_cone_error(form.lower, form.upper, Λ)), atol)
end
