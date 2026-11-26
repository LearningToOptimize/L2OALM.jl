module L2OALMLuxExt

import L2OCore
import Random
using L2OALM:
    L2OALM,
    ALMFormulation,
    ALMMethod,
    ALMTrainer,
    dual_loss,
    dual_target,
    instance_violations,
    map_primal,
    primal_loss,
    update_penalty
using Lux: Lux
using Optimisers: Optimisers
using Lux.Training: TrainState, single_train_step!

const LuxALMTrainer = ALMTrainer{<:TrainState,<:TrainState}

_apply(model, ps, st, P) = first(model(P, ps, Lux.testmode(st)))
_apply(ts::TrainState, P) = _apply(ts.model, ts.parameters, ts.states, P)

"""
    L2OCore.predict_batch(trainer::ALMTrainer{<:TrainState,<:TrainState}, P)

Stacked raw output `[x; λ]` of the primal and dual Lux proxies (test mode), to
be passed to `L2OCore.complete(pipe, P, output)`.
"""
L2OCore.predict_batch(tr::LuxALMTrainer, P::AbstractMatrix) =
    vcat(_apply(tr.primal, P), _apply(tr.dual, P))

function _batches(rng, N::Int, batchsize::Int)
    perm = Random.randperm(rng, N)
    return [perm[i:min(i + batchsize - 1, N)] for i = 1:batchsize:N]
end

function _ordered_batches(N::Int, batchsize::Int)
    return [i:min(i + batchsize - 1, N) for i = 1:batchsize:N]
end

# Learning-rate decay of App. C.3: multiply the phase's rate by `decay` when the
# validation loss exceeds the best value of the phase so far.
function _maybe_decay!(tr, phase::Symbol, ts::TrainState, val, best, decay)
    val <= best && return val
    key = Symbol(phase, :_eta)
    eta = get!(() -> ts.optimizer.eta, tr.options, key) * decay
    tr.options[key] = eta
    tr.options[Symbol(phase, :_decays)] = get(tr.options, Symbol(phase, :_decays), 0) + 1
    Optimisers.adjust!(ts.optimizer_state, eta)
    return best
end

function L2OALM.train!(
    tr::LuxALMTrainer,
    method::ALMMethod,
    form::ALMFormulation,
    inputs::AbstractMatrix;
    outer_iterations::Integer = 1,
    primal_epochs::Integer = 1,
    dual_epochs::Integer = 1,
    batchsize::Integer = size(inputs, 2),
    rng::Random.AbstractRNG = Random.default_rng(),
    ad = Lux.AutoZygote(),
    callback = (trainer, entry) -> nothing,
    violation_reduction = maximum,
    validation::Union{Nothing,AbstractMatrix} = nothing,
    lr_decay::Real = 1,
)
    0 < lr_decay <= 1 || throw(ArgumentError("lr_decay must be in (0, 1]"))
    decaying = validation !== nothing && lr_decay < 1
    T = typeof(tr.ρ)
    N = size(inputs, 2)
    for _ = 1:outer_iterations
        ρ = tr.ρ

        # 1. Primal phase: minimize eq. (5) with the dual proxy frozen.
        dual_ts = tr.dual
        primal_obj = function (model, ps, st, (P, Λ))
            Z, st_new = model(P, ps, st)
            X = map_primal(method, form, Z)
            return primal_loss(method, form, P, X, Λ; ρ = ρ), st_new, (;)
        end
        pts = tr.primal
        primal_mean = zero(T)
        best = T(Inf)
        Λval = decaying ? _apply(dual_ts, validation) : nothing
        for epoch = 1:primal_epochs
            acc, nb = zero(T), 0
            for idx in _batches(rng, N, batchsize)
                P = inputs[:, idx]
                Λ = _apply(dual_ts, P)
                _, ℓ, _, pts = single_train_step!(ad, primal_obj, (P, Λ), pts)
                acc += T(ℓ)
                nb += 1
            end
            epoch == primal_epochs && (primal_mean = acc / max(nb, 1))
            if decaying
                Xv = map_primal(method, form, _apply(pts, validation))
                val = T(primal_loss(method, form, validation, Xv, Λval; ρ = ρ))
                best = _maybe_decay!(tr, :primal, pts, val, best, lr_decay)
            end
        end
        tr.primal = pts

        # 2. Violation measure v_k of eq. (8) over the whole training set
        #    (maximum of the per-instance violations, or another reduction).
        vs = T[]
        for idx in _ordered_batches(N, batchsize)
            P = inputs[:, idx]
            X = map_primal(method, form, _apply(pts, P))
            append!(
                vs,
                Array(instance_violations(method, form, P, X, _apply(dual_ts, P); ρ = ρ)),
            )
        end
        v = isempty(vs) ? zero(T) : T(violation_reduction(vs))

        # 3. Frozen copy λ_k of the dual proxy.
        frozen = (dual_ts.model, deepcopy(dual_ts.parameters), deepcopy(dual_ts.states))

        # 4. Dual phase: regress onto the targets of eq. (6) with eq. (7).
        dual_obj = function (model, ps, st, (P, target))
            Λ, st_new = model(P, ps, st)
            return dual_loss(method, form, Λ, target), st_new, (;)
        end
        dts = dual_ts
        dual_mean = zero(T)
        best = T(Inf)
        Tval = if decaying
            Xv = map_primal(method, form, _apply(pts, validation))
            dual_target(method, form, validation, Xv, _apply(frozen..., validation); ρ = ρ)
        else
            nothing
        end
        for epoch = 1:dual_epochs
            acc, nb = zero(T), 0
            for idx in _batches(rng, N, batchsize)
                P = inputs[:, idx]
                X = map_primal(method, form, _apply(pts, P))
                target = dual_target(method, form, P, X, _apply(frozen..., P); ρ = ρ)
                _, ℓ, _, dts = single_train_step!(ad, dual_obj, (P, target), dts)
                acc += T(ℓ)
                nb += 1
            end
            epoch == dual_epochs && (dual_mean = acc / max(nb, 1))
            if decaying
                val = T(dual_loss(method, form, _apply(dts, validation), Tval))
                best = _maybe_decay!(tr, :dual, dts, val, best, lr_decay)
            end
        end
        tr.dual = dts

        # 5. Penalty rule of eq. (9).
        tr.iteration += 1
        entry = (
            iteration = tr.iteration,
            ρ = ρ,
            violation = v,
            primal_loss = primal_mean,
            dual_loss = dual_mean,
        )
        push!(tr.history, entry)
        tr.ρ = T(update_penalty(method, ρ, v, tr.v_prev))
        tr.v_prev = v
        callback(tr, entry)
    end
    return tr
end

end # module L2OALMLuxExt
