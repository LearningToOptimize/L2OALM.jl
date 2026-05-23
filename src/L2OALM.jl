module L2OALM

using BatchNLPKernels
using ExaModels

using Lux
using LuxCUDA
using Lux.Training
using CUDA
using NNlib: sigmoid_fast
using Random: AbstractRNG
using Statistics
using ChainRules: @ignore_derivatives
using Zygote

using Optimisers

export ALMMethod,
    ALMTrainer, BoundedOutput, FixRefBus, dual_loss, primal_loss, penalty_only_primal_loss,
    analytical_primal_loss, single_train_step!, train!

"""
    BoundedOutput(lvar, uvar)

Parameterless Lux layer that enforces per-variable bounds on the primal network's
output. For variables with finite `(lvar[i], uvar[i])` it applies

    yᵢ = lvar[i] + (uvar[i] − lvar[i]) · sigmoid_fast(zᵢ)

so the output is in `(lvar[i], uvar[i])` asymptotically (sigmoid never reaches the
bounds exactly, but gradient is always nonzero). Variables with an infinite bound
(e.g., voltage angles) pass through unchanged.

Uses logistic sigmoid_fast (not hardsigmoid) so that generators can always recover from
near-zero outputs: hardsigmoid has a dead gradient zone at z < -3 that permanently
traps outputs at the lower bound once entered. Sigmoid'(z) = σ(z)(1-σ(z)) > 0 for
all z, and Adam's gradient normalization restores steps even at deep saturation.
"""
struct BoundedOutput{V<:AbstractVector} <: Lux.AbstractLuxLayer
    lo::V       # safe lower bound: lvar[i] where finite, 0 elsewhere
    width::V    # uvar[i] - lvar[i] where both finite, 0 elsewhere
    mask::V     # 1.0 where both finite, 0.0 elsewhere  (Float for GPU broadcasting)
end

function BoundedOutput(lvar::AbstractVector, uvar::AbstractVector)
    @assert length(lvar) == length(uvar) "lvar and uvar must have the same length"
    T = promote_type(eltype(lvar), eltype(uvar))
    both = isfinite.(lvar) .& isfinite.(uvar)
    lo = T.(ifelse.(both, lvar, zero(T)))
    width = T.(ifelse.(both, uvar .- lvar, zero(T)))
    mask = T.(both)
    return BoundedOutput(lo, width, mask)
end

Lux.initialparameters(::AbstractRNG, ::BoundedOutput) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::BoundedOutput) = NamedTuple()
Lux.parameterlength(::BoundedOutput) = 0
Lux.statelength(::BoundedOutput) = 0

function (b::BoundedOutput)(z::AbstractMatrix, _ps, st::NamedTuple)
    bounded = b.lo .+ b.width .* sigmoid_fast.(z)
    y = b.mask .* bounded .+ (one(eltype(b.mask)) .- b.mask) .* z
    return y, st
end

"""
    FixRefBus(mask)

Parameterless Lux layer that forces reference-bus voltage angle(s) to zero by
multiplying the corresponding rows of the output matrix by zero. This satisfies
the `va[ref] = 0` equality constraint architecturally, removing it from the
Lagrangian penalty and freeing network capacity for the remaining 127 variables.

Construct via `FixRefBus(nvar, ref_bus_idxs)` which builds the mask on CPU; the
mask is transferred to GPU when the Chain is moved with `|> dev`.
"""
struct FixRefBus{V<:AbstractVector} <: Lux.AbstractLuxLayer
    mask::V   # length nvar; 0 at ref bus angle positions, 1 elsewhere
end

function FixRefBus(nvar::Int, ref_bus_idxs)
    mask = ones(Float32, nvar)
    for idx in ref_bus_idxs
        mask[idx] = 0f0
    end
    return FixRefBus(mask)
end

Lux.initialparameters(::AbstractRNG, ::FixRefBus) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::FixRefBus) = NamedTuple()
Lux.parameterlength(::FixRefBus) = 0
Lux.statelength(::FixRefBus) = 0

function (f::FixRefBus)(X::AbstractMatrix, _ps, st::NamedTuple)
    return X .* f.mask, st
end

"""
    AbstractL2OMethod

Abstract type for Learning to Optimize (L2O) methods.
"""
abstract type AbstractL2OMethod end

"""
    AbstractPrimalDualMethod

Abstract type for Primal-Dual Learning to Optimize (L2O) methods.
"""
abstract type AbstractPrimalDualMethod <: AbstractL2OMethod end

"""
    ALMMethod{T<:Real} <: AbstractPrimalDualMethod

Augmented Lagrangian Method (ALM) for primal-dual Learning to Optimize (L2O) methods.
"""
struct ALMMethod{T<:Real} <: AbstractPrimalDualMethod
    batch_model::BatchModel
    max_dual::T
    ρmax::T
    τ::T
    α::T
    num_equal::Int # TODO: There should be a way to get this from the batch model
    ρ_eq_scale::T  # extra penalty multiplier for equality constraints (default 1.0)
    use_analytical_dual::Bool  # apply ALM dual update analytically per gradient step
    use_dual_learning::Bool    # train the dual network; false = frozen dual throughout

    function ALMMethod(batch_model::BatchModel, max_dual::T, ρmax::T, τ::T, α::T, num_equal::Int, ρ_eq_scale::T, use_analytical_dual::Bool, use_dual_learning::Bool) where {T<:Real}
        new{T}(batch_model, max_dual, ρmax, τ, α, num_equal, ρ_eq_scale, use_analytical_dual, use_dual_learning)
    end
end

"""
    ALMMethod(; batch_model, num_equal, max_dual=1e6, ρmax=1e6, τ=0.8, α=10.0, ρ_eq_scale=1.0)

Constructor for the ALM method.

`ρ_eq_scale`: extra multiplier applied to the equality-constraint penalty (and dual
update) relative to the inequality penalty. Values > 1 focus gradient pressure on
power-balance residuals when those equalities are harder to satisfy than the
inequality constraints. Default 1.0 reproduces the standard ALM formulation.
"""
function ALMMethod(;
    batch_model::BatchModel,
    num_equal::Int,
    max_dual::T = 1e6,
    ρmax::T = 1e6,
    τ::T = 0.8,
    α::T = 10.0,
    ρ_eq_scale::T = 1.0,
    use_analytical_dual::Bool = false,
    use_dual_learning::Bool = true,
) where {T<:Real}
    return ALMMethod(batch_model, max_dual, ρmax, τ, α, num_equal, ρ_eq_scale, use_analytical_dual, use_dual_learning)
end

"""
    AbstractL2OTrainer

An abstract type for a structure that holds the training state for Learning to Optimize (L2O) methods.
"""
abstract type AbstractL2OTrainer end

"""
    AbstractPrimalDualTrainer

An abstract type for a structure that holds the training state for Primal-Dual Learning to Optimize (L2O) methods.
"""
abstract type AbstractPrimalDualTrainer <: AbstractL2OTrainer end

"""
    ALMTrainer{T<:Real} <: AbstractPrimalDualTrainer

A structure that holds the training state for the Augmented Lagrangian Method (ALM) for primal-dual Learning to Optimize (L2O) methods.
"""
mutable struct ALMTrainer{T<:Real} <: AbstractPrimalDualTrainer
    primal_model::Lux.Chain
    primal_training_state::Lux.Training.TrainState
    dual_model::Lux.Chain
    dual_training_state::Lux.Training.TrainState
    ρ::T
    prev_dual_training_state::Lux.Training.TrainState
    max_violations::T
    mean_violations::T
    mean_objs::T
    total_loss::T
    dual_loss::T

    function ALMTrainer(
        primal_model::Lux.Chain,
        primal_training_state::Lux.Training.TrainState,
        dual_model::Lux.Chain,
        dual_training_state::Lux.Training.TrainState,
        ρ::T = 1.0,
        prev_dual_training_state::Lux.Training.TrainState = deepcopy(dual_training_state),
        max_violations::T = Inf,
        mean_violations::T = Inf,
        mean_objs::T = Inf,
        total_loss::T = Inf,
        dual_loss::T = Inf,
    ) where {T<:Real}
        new{T}(
            primal_model, primal_training_state, dual_model, dual_training_state, ρ, prev_dual_training_state, 
            max_violations, mean_violations, mean_objs, total_loss, dual_loss
        )
    end
end

function ALMTrainer(;
    primal_model::Lux.Chain,
    primal_training_state::Lux.Training.TrainState,
    dual_model::Lux.Chain,
    dual_training_state::Lux.Training.TrainState,
    ρ::T = 1.0,
) where {T<:Real}
    return ALMTrainer{T}(
        primal_model, primal_training_state, dual_model, dual_training_state, ρ,
    )
end

"""
    dual_loss(method::ALMMethod)

Returns a function that computes the MSE loss for the (dual-)model predicting lagrangian dual variables
from constraint evaluations and the last dual predictions.
Target is calculated using the augmented lagrangian method. 
Target dual variables are clipped from zero to `max_dual`.
"""
function dual_loss(method::ALMMethod)
    bm = method.batch_model
    max_dual = method.max_dual
    num_equal = method.num_equal
    ρ_eq_scale = method.ρ_eq_scale
    return (dual_model, ps_dual, st_dual, data) -> begin
        Θ, trainer = data
        ρ = trainer.ρ
        prev_dual_training_state = trainer.prev_dual_training_state
        primal_training_state = trainer.primal_training_state

        # Get current dual predictions
        dual_hat, st_dual_new = dual_model(Θ, ps_dual, st_dual)

        # Get previous dual predictions
        dual_hat_k, _ = @ignore_derivatives dual_model(Θ, prev_dual_training_state.parameters, prev_dual_training_state.states)

        # Separate bound and equality constraints (frozen primal — no gradients needed here)
        X̂_dual, _ = @ignore_derivatives trainer.primal_model(Θ, primal_training_state.parameters, primal_training_state.states)
        gh = @ignore_derivatives BNK.constraints!(bm, X̂_dual, Θ)
        gh_bound = gh[1:end-num_equal, :]
        gh_equal = gh[end-num_equal+1:end, :]
        dual_hat_bound = dual_hat_k[1:end-num_equal, :]
        dual_hat_equal = dual_hat_k[end-num_equal+1:end, :]

        # ALM dual targets:
        #   μ_{k+1} ← max(μ_k + ρ·g, 0)         (ineq: one-sided clip)
        #   λ_{k+1} ← λ_k + ρ·ρ_eq_scale·h      (eq: symmetric; scale boosts equality pressure)
        # The outer clamp is a numerical safeguard against unbounded growth.
        dual_target = vcat(
            clamp.(dual_hat_bound + ρ .* gh_bound, zero(max_dual), max_dual),
            clamp.(dual_hat_equal + (ρ * ρ_eq_scale) .* gh_equal, -max_dual, max_dual),
        )

        loss = mean((dual_hat .- dual_target) .^ 2)
        return loss, st_dual_new, (dual_loss = loss,)
    end
end

"""
    primal_loss(method::ALMMethod)

Returns a function that computes the augmented lagrangian primal loss
from current dual predictions for the batch model `bm` under parameters `Θ`.
"""
function primal_loss(method::ALMMethod)
    bm = method.batch_model
    num_equal = method.num_equal
    ρ_eq_scale = method.ρ_eq_scale
    return (model, ps, st, data) -> begin
        Θ, trainer = data
        ρ = trainer.ρ
        num_s = size(Θ, 2)

        X̂, st_new = model(Θ, ps, st)

        # Frozen dual predictions for this primal step.
        dual_hat, _ = @ignore_derivatives trainer.dual_model(
            Θ, trainer.dual_training_state.parameters, trainer.dual_training_state.states)

        # Paper PDL formulation (Park & Van Hentenryck 2023):
        #   L_ρ(y, μ, λ) = f(y) + μᵀg(y) + λᵀh(y) + (ρ/2)·(‖max(g,0)‖² + ‖h‖²)
        # Lagrangian linear term uses RAW signed g, h from BNK.constraints!
        # (`max(g,0)`/`|h|` from all_violations! is the wrong quantity here:
        # μ ≥ 0 with g_i ≤ 0 should contribute *negatively* so the primal keeps
        # inequalities slack rather than just non-positive).
        # Penalty term squares Vc, which BNK.all_violations! already computes as
        # max(g,0) for inequalities and |h| for equalities — squaring gives the
        # paper's max(g,0)² + h² exactly.
        # ρ_eq_scale > 1 applies extra pressure on equality violations specifically,
        # matching the dual_loss update which also uses ρ·ρ_eq_scale for equalities.
        objs = BNK.objective!(bm, X̂, Θ)
        gh   = BNK.constraints!(bm, X̂, Θ)

        # Split by ordering: inequalities first, equalities last (enforced in power.jl).
        g = gh[1:end-num_equal, :]
        h = gh[end-num_equal+1:end, :]
        # Clamp inequality multipliers to μ ≥ 0: the dual network output is unconstrained
        # so it can predict negative values, which would flip the sign of the μᵀg term and
        # push the primal TOWARD constraint violations. Clamping restores KKT sign correctness.
        μ = max.(dual_hat[1:end-num_equal, :], zero(eltype(dual_hat)))
        λ = dual_hat[end-num_equal+1:end, :]

        # Compute violations from gh directly — avoids a second BNK.constraints! call
        # through all_violations! (which shares bm.cons_out and corrupts the rrule view).
        Vc_ineq = max.(g, zero(eltype(g)))
        Vc_eq   = abs.(h)
        Vc      = vcat(Vc_ineq, Vc_eq)

        lagrangian_term = (sum(μ .* g) + sum(λ .* h)) / num_s
        penalty_term    = (ρ / 2) * (sum(Vc_ineq .^ 2) + ρ_eq_scale * sum(Vc_eq .^ 2)) / num_s
        total_loss      = mean(objs) + lagrangian_term + penalty_term

        return total_loss,
        st_new,
        (
            total_loss      = total_loss,
            mean_violations = mean(Vc),
            max_violation   = maximum(Vc),
            max_bound_violation = zero(eltype(Vc)),
            mean_objs       = mean(objs),
        )
    end
end

"""
    reconcile_primal(::AbstractL2OTrainer, batch_states::Vector{NamedTuple})

Reconciles the state of the primal model after processing a batch of data.
This function computes the maximum violation, mean violations, mean objectives, and total loss
from the batch states.
"""
function reconcile_primal(::AbstractL2OTrainer, batch_states::Vector{NamedTuple})
    # Use mean of mean_violations (not max-of-max) as the ρ-update signal.
    # max-of-max is dominated by noisy hard batches and causes ρ to escalate
    # before violations have genuinely plateaued, collapsing generators.
    mean_violations = mean([s.mean_violations for s in batch_states])
    mean_objs = mean([s.mean_objs for s in batch_states])
    mean_loss = mean([s.total_loss for s in batch_states])
    return (;
        max_violation = mean_violations,   # proxy for ρ criterion: bump only when mean stalls
        mean_violations = mean_violations,
        mean_objs = mean_objs,
        total_loss = mean_loss,
    )
end

"""
    reconcile_dual(::AbstractL2OTrainer, batch_states::Vector{NamedTuple})

Reconciles the state of the dual model after processing a batch of data.
This function computes the mean dual loss from the batch states.
"""
function reconcile_dual(::AbstractL2OTrainer, batch_states::Vector{NamedTuple})
    isempty(batch_states) && return (dual_loss = 0.0,)
    return (dual_loss = mean([s.dual_loss for s in batch_states]),)
end

"""
    update_trainer!(method::ALMMethod, trainer::ALMTrainer,
                   primal_state::NamedTuple, dual_state::NamedTuple)

Update the hyperparameters and states in the ALM algorithm.
This function increases ρ by a factor of α if the new maximum violation exceeds τ times the previous maximum violation.
"""
function update_trainer!(
    method::ALMMethod,
    trainer::ALMTrainer,
    primal_state::NamedTuple,
    dual_state::NamedTuple,
)
    # Update primal state
    new_max_violations = primal_state.max_violation
    trainer.mean_violations = primal_state.mean_violations
    trainer.mean_objs = primal_state.mean_objs
    trainer.total_loss = primal_state.total_loss
    # Update ρ if necessary
    if new_max_violations > method.τ * trainer.max_violations
        trainer.ρ = min(method.ρmax, trainer.ρ * method.α)
    end
    trainer.max_violations = new_max_violations

    # Update dual state
    trainer.dual_loss = dual_state.dual_loss
    if method.use_dual_learning
        trainer.prev_dual_training_state = deepcopy(trainer.dual_training_state)
    end

    return
end

"""
    _stopping_criterion(iter::Int, method::AbstractL2OMethod, trainer::AbstractL2OTrainer)

Default stopping criterion for the L2O methods.
This function checks if the number of iterations has reached a predefined limit (100).
"""
function _stopping_criterion(iter::Int, ::M, ::N) where {M<:AbstractL2OMethod, N<:AbstractL2OTrainer}
    return iter >= 2 ? true : false
end

"""
    primal_stopping_criterion(iter::Int, method::AbstractL2OMethod, trainer::AbstractL2OTrainer)

Default stopping criterion for primal learning methods.
This function checks if the number of iterations has reached a predefined limit (100).
"""
function primal_stopping_criterion(iter::Int,  method::M, trainer::N) where {M<:AbstractL2OMethod, N<:AbstractL2OTrainer}
    return _stopping_criterion(iter, method, trainer)
end

"""
    dual_stopping_criterion(iter::Int, method::AbstractL2OMethod, trainer::AbstractL2OTrainer)

Default stopping criterion for dual learning methods.
This function checks if the number of iterations has reached a predefined limit (100).
"""
function dual_stopping_criterion(iter::Int,  method::M, trainer::N) where {M<:AbstractL2OMethod, N<:AbstractL2OTrainer}
    return _stopping_criterion(iter, method, trainer)
end

"""
    single_train_step!(
    method::AbstractPrimalDualMethod,
    trainer::AbstractPrimalDualTrainer,
    data,
)

Performs a single training step for the primal-dual method.
This function loops through the primal learning method until the stopping criterion is met
with the dual model fixed, then the inverse is done with the dual learning method and finally
updates the trainer state.
"""
function single_train_step!(
    method::ALMMethod,
    trainer::ALMTrainer,
    data;
    L_primal::Int = 1,
    L_dual::Int   = 1,
)
    _primal_loss = method.use_analytical_dual ? analytical_primal_loss(method) : primal_loss(method)
    train_state_primal = trainer.primal_training_state
    train_state_dual   = trainer.dual_training_state

    # ----- primal phase: exactly L_primal gradient steps, dual frozen -----
    primal_states = NamedTuple[]
    primal_step = 0
    while primal_step < L_primal
        for θ in data
            primal_step >= L_primal && break
            _, _, stats, train_state_primal = Training.single_train_step!(
                AutoZygote(), _primal_loss, (θ, trainer), train_state_primal,
            )
            push!(primal_states, stats)
            primal_step += 1
        end
    end
    trainer.primal_training_state = train_state_primal
    current_state_primal = reconcile_primal(trainer, primal_states)

    # ----- dual phase: exactly L_dual gradient steps, primal frozen -----
    # Skipped entirely when use_dual_learning = false (dual network stays frozen).
    dual_states = NamedTuple[]
    if method.use_dual_learning
        _dual_loss = dual_loss(method)
        dual_step = 0
        while dual_step < L_dual
            for θ in data
                dual_step >= L_dual && break
                _, _, stats, train_state_dual = Training.single_train_step!(
                    AutoZygote(), _dual_loss, (θ, trainer), train_state_dual,
                )
                push!(dual_states, stats)
                dual_step += 1
            end
        end
        trainer.dual_training_state = train_state_dual
    end
    update_trainer!(method, trainer, current_state_primal, reconcile_dual(trainer, dual_states))
    return
end

"""
    penalty_only_primal_loss(method::ALMMethod)

Closure that computes a *penalty-only* primal loss (no Lagrangian linear term):

    L(y) = f(y) + (ρ/2) · ‖V(y)‖²

Used for the primal warm-start phase before the alternating loop, so the primal
network finds a feasible basin against fixed-zero multipliers before the dual
starts producing nonzero multipliers. Without this, the first few outer
iterations can be unstable because the dual fits multipliers against random
primal predictions.
"""
function penalty_only_primal_loss(method::ALMMethod)
    bm = method.batch_model
    num_equal = method.num_equal
    ρ_eq_scale = method.ρ_eq_scale
    return (model, ps, st, data) -> begin
        Θ, trainer = data
        ρ = trainer.ρ
        num_s = size(Θ, 2)
        X̂, st_new = model(Θ, ps, st)
        objs = BNK.objective!(bm, X̂, Θ)
        # Use BNK.constraints! (which has an rrule) so that penalty gradients flow
        # back through to the network weights. BNK.all_violations! has no rrule and
        # would give zero penalty gradient, collapsing generators to pmin.
        gh = BNK.constraints!(bm, X̂, Θ)
        g  = gh[1:end-num_equal, :]
        h  = gh[end-num_equal+1:end, :]
        Vc_ineq = max.(g, zero(eltype(g)))
        Vc_eq   = abs.(h)
        Vc      = vcat(Vc_ineq, Vc_eq)
        penalty_term = (ρ / 2) * (sum(Vc_ineq .^ 2) + ρ_eq_scale * sum(Vc_eq .^ 2)) / num_s
        total_loss = mean(objs) + penalty_term
        return total_loss, st_new, (
            total_loss      = total_loss,
            mean_violations = mean(Vc),
            max_violation   = maximum(Vc),
            max_bound_violation = zero(eltype(Vc)),
            mean_objs       = mean(objs),
        )
    end
end

"""
    analytical_primal_loss(method::ALMMethod)

Primal loss that applies the ALM dual update analytically at every gradient step,
eliminating the dual tracking gap entirely.

Instead of using the dual network's output directly (which lags behind the correct
multipliers by a 4000× factor at ρmax=10000), this computes the analytically corrected
multipliers at each step:

    μ_eff = max(μ_net(θ) + ρ·g(y,θ),  0)
    λ_eff = clamp(λ_net(θ) + ρ·h(y,θ), -max_dual, max_dual)

Then uses these in the augmented Lagrangian:

    L = f(y) + μ_eff·g(y) + λ_eff·h(y)

The penalty term is implicit: μ_eff·g = (μ_net + ρg)·g includes a ρg² term for
violated constraints, equivalent to the standard ALM penalty. The dual network still
trains in parallel (providing a warm-start for test-time generalization) but the primal
gradient is no longer bottlenecked by slow dual convergence.
"""
function analytical_primal_loss(method::ALMMethod)
    bm = method.batch_model
    num_equal = method.num_equal
    ρ_eq_scale = method.ρ_eq_scale
    max_dual = method.max_dual
    return (model, ps, st, data) -> begin
        Θ, trainer = data
        ρ = trainer.ρ
        num_s = size(Θ, 2)

        X̂, st_new = model(Θ, ps, st)

        objs = BNK.objective!(bm, X̂, Θ)
        gh   = BNK.constraints!(bm, X̂, Θ)

        g = gh[1:end-num_equal, :]
        h = gh[end-num_equal+1:end, :]

        # Dual network provides a warm-start base; analytically correct by ρ·gh.
        # @ignore_derivatives: gradient does NOT flow through dual net params,
        # but DOES flow through gh (computed from X̂ above), providing the correct
        # ρ·g and ρ·h gradient terms without any tracking lag.
        dual_hat, _ = @ignore_derivatives trainer.dual_model(
            Θ, trainer.dual_training_state.parameters, trainer.dual_training_state.states)

        μ_base = dual_hat[1:end-num_equal, :]
        λ_base = dual_hat[end-num_equal+1:end, :]

        # Analytically corrected multipliers (one full ALM update applied immediately)
        μ = clamp.(μ_base .+ ρ .* g,                  zero(max_dual), max_dual)
        λ = clamp.(λ_base .+ (ρ * ρ_eq_scale) .* h,  -max_dual,      max_dual)

        Vc_ineq = max.(g, zero(eltype(g)))
        Vc_eq   = abs.(h)
        Vc      = vcat(Vc_ineq, Vc_eq)

        lagrangian_term = (sum(μ .* g) + sum(λ .* h)) / num_s
        total_loss      = mean(objs) + lagrangian_term

        return total_loss, st_new, (
            total_loss      = total_loss,
            mean_violations = mean(Vc),
            max_violation   = maximum(Vc),
            max_bound_violation = zero(eltype(Vc)),
            mean_objs       = mean(objs),
        )
    end
end

"""
    train!(method, trainer, data;
        K=10, L_primal=500, L_dual=500, warmup_epochs=4,
        lr_primal=1e-4, lr_dual=1e-4, lr_decay=0.99,
        eval_fn=nothing)

Top-level training loop matching the PDL paper schedule.

- `K`: number of outer alternating iterations.
- `L_primal` / `L_dual`: inner epochs (passes through `data`) per phase.
- `warmup_epochs`: number of penalty-only primal epochs before the alternating
  loop. Set to 0 to disable.
- `lr_primal`, `lr_dual`: starting Adam learning rates. The trainer's existing
  TrainStates are mutated via `Optimisers.adjust!` so this overrides whatever
  was set at construction.
- `lr_decay`: multiplicative decay applied when validation loss worsens.
- `eval_fn(iter, trainer) -> Real | nothing`: optional validation callback. If
  it returns a Real that's larger than the previous, both lrs are multiplied
  by `lr_decay`.
"""
function train!(
    method::ALMMethod, trainer::ALMTrainer, data;
    K::Int = 10, L_primal::Int = 500, L_dual::Int = 500,
    warmup_epochs::Int = 4,
    lr_primal::Real = 1e-4, lr_dual::Real = 1e-4,
    lr_decay::Real = 0.99,
    eval_fn = nothing,
)
    Optimisers.adjust!(trainer.primal_training_state.optimizer_state, lr_primal)
    Optimisers.adjust!(trainer.dual_training_state.optimizer_state,   lr_dual)
    cur_lr_primal = lr_primal
    cur_lr_dual   = lr_dual

    # ----- warm-start: penalty-only primal (warmup_epochs = gradient steps) -----
    # Run at ρmax (not ρ_init) so the penalty is strong enough to push the primal
    # toward feasibility before the dual starts producing multipliers. At ρ_init=10
    # the penalty gradient is ~1000× too weak to reduce power-balance violations.
    if warmup_epochs > 0
        warmup_loss = penalty_only_primal_loss(method)
        ts_primal = trainer.primal_training_state
        ρ_saved = trainer.ρ
        trainer.ρ = method.ρmax     # full penalty during warmup
        warmup_step = 0
        while warmup_step < warmup_epochs
            for θ in data
                warmup_step >= warmup_epochs && break
                _, _, _, ts_primal = Training.single_train_step!(
                    AutoZygote(), warmup_loss, (θ, trainer), ts_primal,
                )
                warmup_step += 1
            end
        end
        trainer.primal_training_state = ts_primal
        trainer.ρ = ρ_saved         # restore initial ρ for ALM schedule
    end

    # ----- alternating primal/dual loop with optional lr decay -----
    last_val = typemax(Float64)
    for outer in 1:K
        single_train_step!(method, trainer, data; L_primal = L_primal, L_dual = L_dual)
        if eval_fn !== nothing
            v = eval_fn(outer, trainer)
            if v isa Real
                if v > last_val
                    cur_lr_primal *= lr_decay
                    cur_lr_dual   *= lr_decay
                    Optimisers.adjust!(trainer.primal_training_state.optimizer_state, cur_lr_primal)
                    Optimisers.adjust!(trainer.dual_training_state.optimizer_state,   cur_lr_dual)
                end
                last_val = v
            end
        end
    end
    return
end

end
