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
    map_primal,
    primal_loss,
    update_penalty,
    violation_measure
using Lux: Lux
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
)
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
        end
        tr.primal = pts

        # 2. Violation measure v_k of eq. (8) over the whole training set.
        v = zero(T)
        for idx in _ordered_batches(N, batchsize)
            P = inputs[:, idx]
            X = map_primal(method, form, _apply(pts, P))
            v = max(v, T(violation_measure(method, form, P, X, _apply(dual_ts, P); ρ = ρ)))
        end

        # 3. Frozen copy λ_k of the dual proxy.
        frozen = (dual_ts.model, deepcopy(dual_ts.parameters), deepcopy(dual_ts.states))

        # 4. Dual phase: regress onto the targets of eq. (6) with eq. (7).
        dual_obj = function (model, ps, st, (P, target))
            Λ, st_new = model(P, ps, st)
            return dual_loss(method, form, Λ, target), st_new, (;)
        end
        dts = dual_ts
        dual_mean = zero(T)
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
