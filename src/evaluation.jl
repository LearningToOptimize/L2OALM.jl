"""
    dual_sign_violation(problem::ParametricProblem, prediction) -> Vector
    dual_sign_violation(form::ALMFormulation, prediction) -> Vector

Per instance, the largest violation of the sign conditions of the dual cone by
the multipliers of `prediction` (a [`Prediction`](@ref L2OCore.Prediction) with
a dual block, or a `num_duals × B` matrix / length-`num_duals` vector), in the
`JuMP.dual` convention: `y ≥ 0` for `GreaterThan`/`Nonnegatives` rows,
`y ≤ 0` for `LessThan`/`Nonpositives` rows, any sign for equalities and
two-sided intervals.

[`ALMMethod`](@ref) makes this zero by construction
([`project_multipliers`](@ref) in `L2OCore.complete`). Sign feasibility alone
does not make the multipliers dual feasible: PDL multipliers do not satisfy
Lagrangian stationarity, so they need not give a valid bound through
`L2OCore.dual_bound`.
"""
function dual_sign_violation(problem::ParametricProblem, prediction)
    d = ALMDecomposition(problem)
    return _sign_error(d.lower, d.upper, from_jump_duals(_dualbatch(prediction)))
end

function dual_sign_violation(form::ALMFormulation, prediction)
    return _sign_error(form.lower, form.upper, from_jump_duals(_dualbatch(prediction)))
end

function _sign_error(lower, upper, Λ)
    # Λ in the L2OALM convention; violation of λᵢ ≤ 0 (no upper bound) and
    # λᵢ ≥ 0 (no lower bound).
    E =
        ifelse.(isfinite.(upper), zero(eltype(Λ)), max.(Λ, 0)) .+
        ifelse.(isfinite.(lower), zero(eltype(Λ)), max.(.-Λ, 0))
    return size(E, 1) == 0 ? zeros(eltype(E), size(E, 2)) :
           collect(vec(maximum(E; dims = 1)))
end

_dualbatch(Y::AbstractVector) = reshape(Y, :, 1)
_dualbatch(Y::AbstractMatrix) = Y
_dualbatch(prediction::Prediction) =
    prediction.dual === nothing ?
    throw(ArgumentError("dual_sign_violation needs a dual prediction")) :
    _dualbatch(prediction.dual)
