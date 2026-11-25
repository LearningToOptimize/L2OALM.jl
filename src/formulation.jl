"""
    ALMFormulation(evaluator, lower, upper, xlower, xupper, equality; sense = 1, bound = 0)
    ALMFormulation(pipe::L2OPipeline)

The problem as the augmented Lagrangian method sees it:

```math
\\min_x \\; s \\, f(x; p) \\quad \\text{s.t.} \\quad l \\le c(x; p) \\le u,
```

where `c(x; p) = L2OCore.constraint_values(evaluator, X, P)` stacks every
constraint row (variable bounds included), `l = lower` and `u = upper` may
contain `±Inf`, and `s = sense` is `+1` for minimization and `-1` for
maximization. `xlower`/`xupper` are the variable bounds (used by bound maps),
`equality` is `1` on rows with `l == u` and `0` elsewhere, and `bound` is `1`
on rows that are variable bounds (`c_i(x) = x_j`) and `0` elsewhere (default:
no bound rows). Bound rows get no multiplier when the bound map enforces the
bounds (see [`enforces_bounds`](@ref)), as in the paper, where the variable
domain `Y` is handled by the network architecture.

All vectors share one array type so that the formulation lives on the same
device as the evaluator: CPU `Vector`s for JuMP problems, device vectors for an
ExaModels/BatchNLPKernels evaluator built on a GPU backend.

Constructors:

  - `ALMFormulation(pipe)` for an `L2OCore.L2OPipeline` whose method is an
    [`ALMMethod`](@ref): rows are `L2OCore.dual_layout(pipe.problem)`, the
    evaluator is `pipe.evaluator`.
  - `ALMFormulation(bm::BatchNLPKernels.BatchModel)` (extension
    `L2OALMBatchNLPKernelsExt`): rows are the ExaModel constraints followed by
    the variables (as bound rows).

**Extension point.** Any `L2OCore.AbstractEvaluator` that implements
`batch_objective` and `constraint_values` (differentiably, for training) can be
used; build the formulation with the positional constructor.

**Multiplier convention.** Inside L2OALM a multiplier `λᵢ` of row `i` lives in
the normal cone of `[lᵢ, uᵢ]` at `cᵢ(x)`: `λᵢ ≥ 0` pushes against the upper
bound, `λᵢ ≤ 0` against the lower bound, and the Lagrangian is
`s f(x) + λᵀ c(x)` (up to constants). This is the paper's convention after
writing `g(x) = c(x) - u ≤ 0` (multiplier `μ = λ ≥ 0`) and `h(x) = c(x) - b`
(multiplier `λ`). `JuMP.dual` returns `-λ`; see [`to_jump_duals`](@ref).
"""
struct ALMFormulation{E<:AbstractEvaluator,V<:AbstractVector{<:Real},T<:Real}
    evaluator::E
    lower::V
    upper::V
    xlower::V
    xupper::V
    equality::V
    bound::V
    sense::T
    function ALMFormulation(
        evaluator::E,
        lower::V,
        upper::V,
        xlower::V,
        xupper::V,
        equality::V;
        sense::Real = 1,
        bound::Union{Nothing,V} = nothing,
    ) where {E<:AbstractEvaluator,V<:AbstractVector{<:Real}}
        m = length(lower)
        bound = bound === nothing ? fill!(similar(lower), 0) : bound
        (length(upper) == m && length(equality) == m && length(bound) == m) || throw(
            DimensionMismatch("lower, upper, equality and bound must have the same length"),
        )
        length(xlower) == length(xupper) ||
            throw(DimensionMismatch("xlower and xupper must have the same length"))
        sense in (1, -1) || throw(ArgumentError("sense must be +1 (min) or -1 (max)"))
        T = eltype(V)
        return new{E,V,T}(
            evaluator,
            lower,
            upper,
            xlower,
            xupper,
            equality,
            bound,
            T(sense),
        )
    end
end

"""
    num_rows(form::ALMFormulation) -> Int

Number of constraint rows (the length of a multiplier vector).
"""
num_rows(form::ALMFormulation) = length(form.lower)

L2OCore.num_decision_variables(form::ALMFormulation) = length(form.xlower)

Base.show(io::IO, f::ALMFormulation) = print(
    io,
    "ALMFormulation(",
    nameof(typeof(f.evaluator)),
    "; ",
    num_rows(f),
    " rows, ",
    length(f.xlower),
    " variables, ",
    f.sense > 0 ? "min" : "max",
    ")",
)

# ---------------------------------------------------------------------------
# Rows of a JuMP-based ParametricProblem

const _MOI = JuMP.MOI

_box(s::_MOI.EqualTo) = (s.value, s.value)
_box(s::_MOI.GreaterThan) = (s.lower, Inf)
_box(s::_MOI.LessThan) = (-Inf, s.upper)
_box(s::_MOI.Interval) = (s.lower, s.upper)
_box(::_MOI.Zeros) = (0.0, 0.0)
_box(::_MOI.Nonnegatives) = (0.0, Inf)
_box(::_MOI.Nonpositives) = (-Inf, 0.0)
_box(s) = throw(
    ArgumentError(
        "ALMMethod supports only element-wise (box) constraint sets " *
        "(EqualTo, GreaterThan, LessThan, Interval, Zeros, Nonnegatives, " *
        "Nonpositives); got $(typeof(s)). Reformulate conic constraints with " *
        "smooth nonlinear constraints first.",
    ),
)

function _variable_box(v::JuMP.VariableRef)
    JuMP.is_fixed(v) && return (JuMP.fix_value(v), JuMP.fix_value(v))
    l = JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf
    u = JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf
    return (l, u)
end

"""
    ALMDecomposition

What an [`ALMMethod`](@ref) predicts for a `ParametricProblem`: all primal
variables followed by one multiplier per constraint row (in
`L2OCore.dual_layout` order), together with the row and variable bounds of the
problem, computed once by `L2OCore.decompose`.
"""
struct ALMDecomposition{T<:Real} <: L2OCore.AbstractDecomposition
    nprimal::Int
    ndual::Int
    lower::Vector{T}
    upper::Vector{T}
    xlower::Vector{T}
    xupper::Vector{T}
    equality::Vector{T}
    bound::Vector{T}
    sense::T
end

L2OCore.output_dimension(d::ALMDecomposition) = d.nprimal + d.ndual

function ALMDecomposition(problem::ParametricProblem; T::Type{<:Real} = Float64)
    m = L2OCore.num_duals(problem)
    lower = fill(T(-Inf), m)
    upper = fill(T(Inf), m)
    bound = zeros(T, m)
    for (c, rows) in zip(L2OCore.constraint_refs(problem), L2OCore.dual_layout(problem))
        obj = JuMP.constraint_object(c)
        l, u = _box(JuMP.moi_set(obj))
        lower[rows] .= l
        upper[rows] .= u
        isbound =
            obj.func isa JuMP.AbstractVariableRef ||
            obj.func isa AbstractVector{<:JuMP.AbstractVariableRef}
        bound[rows] .= isbound
    end
    vars = L2OCore.decision_variables(problem)
    boxes = map(_variable_box, vars)
    xlower = T[first(b) for b in boxes]
    xupper = T[last(b) for b in boxes]
    equality = T.(lower .== upper)
    sense = JuMP.objective_sense(problem.model) == _MOI.MAX_SENSE ? -one(T) : one(T)
    return ALMDecomposition{T}(
        length(vars),
        m,
        lower,
        upper,
        xlower,
        xupper,
        equality,
        bound,
        sense,
    )
end

function ALMFormulation(pipe::L2OPipeline)
    d = pipe.decomposition
    d isa ALMDecomposition || throw(
        ArgumentError(
            "ALMFormulation(pipe) needs a pipeline whose method is an ALMMethod " *
            "(got decomposition $(nameof(typeof(d))))",
        ),
    )
    return ALMFormulation(
        pipe.evaluator,
        d.lower,
        d.upper,
        d.xlower,
        d.xupper,
        d.equality;
        sense = d.sense,
        bound = d.bound,
    )
end

ALMFormulation(form::ALMFormulation) = form
