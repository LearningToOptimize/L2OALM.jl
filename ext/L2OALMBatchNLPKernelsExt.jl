module L2OALMBatchNLPKernelsExt

import L2OCore
using L2OALM: L2OALM, ALMFormulation
using BatchNLPKernels: BatchNLPKernels, BatchModel

"""
    BatchModelEvaluator(bm::BatchModel)

`L2OCore.AbstractEvaluator` backed by a BatchNLPKernels `BatchModel` (an
ExaModel with batch buffers). Rows of `constraint_values` are the ExaModel
constraints followed by the variables (bound rows). Evaluation runs as
KernelAbstractions kernels on the ExaModel's backend (CPU or GPU) and is
differentiable through the ChainRules rules of BatchNLPKernels; build the
`BatchModel` with `BatchModelConfig(:viol_grad)` (or `:gradients`/`:full`)
for training.
"""
struct BatchModelEvaluator{B<:BatchModel} <: L2OCore.AbstractEvaluator
    bm::B
end

Base.show(io::IO, ev::BatchModelEvaluator) = print(
    io,
    "BatchModelEvaluator(",
    ev.bm.model.meta.nvar,
    " variables, ",
    ev.bm.model.meta.ncon,
    " constraints, batch ≤ ",
    ev.bm.batch_size,
    ")",
)

L2OCore.batch_objective(ev::BatchModelEvaluator, X::AbstractMatrix, P::AbstractMatrix) =
    BatchNLPKernels.objective!(ev.bm, X, P)

# `constraints!` returns a view of an internal buffer; `vcat` copies it.
L2OCore.constraint_values(ev::BatchModelEvaluator, X::AbstractMatrix, P::AbstractMatrix) =
    vcat(BatchNLPKernels.constraints!(ev.bm, X, P), X)

function L2OCore.row_violation(
    ev::BatchModelEvaluator,
    X::AbstractMatrix,
    P::AbstractMatrix,
)
    meta = ev.bm.model.meta
    C = L2OCore.constraint_values(ev, X, P)
    l, u = vcat(meta.lcon, meta.lvar), vcat(meta.ucon, meta.uvar)
    return max.(l .- C, 0) .+ max.(C .- u, 0)
end

function L2OCore.equality_rows(ev::BatchModelEvaluator)
    meta = ev.bm.model.meta
    return BitVector(Array(vcat(meta.lcon .== meta.ucon, meta.lvar .== meta.uvar)))
end

function L2OCore.bound_rows(ev::BatchModelEvaluator)
    meta = ev.bm.model.meta
    return BitVector(vcat(falses(meta.ncon), trues(meta.nvar)))
end

"""
    ALMFormulation(bm::BatchNLPKernels.BatchModel)

Formulation of an ExaModels problem for [`L2OALM.ALMMethod`](@ref): rows are the
constraints `lcon ≤ c(x; θ) ≤ ucon` followed by the variable bounds
`lvar ≤ x ≤ uvar`; all vectors live on the ExaModel's device.
"""
function L2OALM.ALMFormulation(bm::BatchModel)
    meta = bm.model.meta
    lower = vcat(meta.lcon, meta.lvar)
    upper = vcat(meta.ucon, meta.uvar)
    T = eltype(lower)
    return ALMFormulation(
        BatchModelEvaluator(bm),
        lower,
        upper,
        copy(meta.lvar),
        copy(meta.uvar),
        T.(lower .== upper);
        sense = meta.minimize ? 1 : -1,
    )
end

end # module L2OALMBatchNLPKernelsExt
