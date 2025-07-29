module L2OALM

using BatchNLPKernels
using ExaModels

using Lux
using LuxCUDA
using Lux.Training
using CUDA
using Statistics
using ChainRules: @ignore_derivatives

export ALMMethod,
    ALMTrainer, dual_loss, primal_loss, single_train_step!

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

    function ALMMethod(batch_model::BatchModel, max_dual::T, ρmax::T, τ::T, α::T, num_equal::Int) where {T<:Real}
        new{T}(batch_model, max_dual, ρmax, τ, α, num_equal)
    end
end

"""
    ALMMethod(; batch_model::BatchModel, num_equal::Int, max_dual::T = 1e6, ρmax::T = 1e6, τ::T = 0.8, α::T = 10.0)

Constructor for the Augmented Lagrangian Method (ALM) for primal-dual Learning to Optimize (L2O) methods.
This function initializes the ALM method with a batch model, maximum dual variable values, maximum learning rate `ρ`, 
threshold for the parameter updater `τ`, meta learning rate `α`, and the number of equality constraints.
"""
function ALMMethod(;
    batch_model::BatchModel,
    num_equal::Int,
    max_dual::T = 1e6,
    ρmax::T = 1e6,
    τ::T = 0.8,
    α::T = 10.0,
) where {T<:Real}
    return ALMMethod(batch_model, max_dual, ρmax, τ, α, num_equal)
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
    return (dual_model, ps_dual, st_dual, data) -> begin
        Θ, trainer = data
        ρ = trainer.ρ
        prev_dual_training_state = trainer.prev_dual_training_state
        primal_training_state = trainer.primal_training_state

        # Get current dual predictions
        dual_hat, st_dual_new = dual_model(Θ, ps_dual, st_dual)

        # Get previous dual predictions
        dual_hat_k, _ = @ignore_derivatives dual_model(Θ, prev_dual_training_state.parameters, prev_dual_training_state.states)

        # Separate bound and equality constraints
        # # Forward pass for primal model
        X̂, _ = trainer.primal_model(θ, primal_training_state.parameters, primal_training_state.states)
        gh = constraints!(bm, X̂, Θ)

        @ignore_derivatives gh = trainer.constraints(Θ, trainer.ps_constraints, trainer.st_constraints)
        gh_bound = gh[1:end-num_equal, :]
        gh_equal = gh[end-num_equal+1:end, :]
        dual_hat_bound = dual_hat_k[1:end-num_equal, :]
        dual_hat_equal = dual_hat_k[end-num_equal+1:end, :]

        # Target for dual variables
        dual_target = vcat(
            min.(max.(dual_hat_bound + ρ .* gh_bound, 0), max_dual),
            min.(dual_hat_equal + ρ .* gh_equal, max_dual),
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
    return (model, ps, st, data) -> begin
        Θ, trainer = data
        ρ = trainer.ρ
        num_s = size(Θ, 2)

        # Forward pass for prediction
        X̂, st_new = model(Θ, ps, st)

        # Get current dual predictions
        dual_hat, _ = @ignore_derivatives trainer.dual_model(Θ, trainer.dual_training_state.parameters, trainer.dual_training_state.states)

        # Calculate violations and objectives
        objs = BNK.objective!(bm, X̂, Θ)
        # gh = constraints!(bm, X̂, Θ)
        Vc, Vb = BNK.all_violations!(bm, X̂, Θ)
        V = vcat(Vb, Vc)
        total_loss = (
            sum(abs.(dual_hat .* V)) / num_s + ρ / 2 * sum((V) .^ 2) / num_s + mean(objs)
        )

        return total_loss,
        st_new,
        (
            total_loss = total_loss,
            mean_violations = mean(V),
            max_violation = maximum(V),
            mean_objs = mean(objs),
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
    max_violation = maximum([s.max_violation for s in batch_states])
    mean_violations = mean([s.mean_violations for s in batch_states])
    mean_objs = mean([s.mean_objs for s in batch_states])
    mean_loss = mean([s.total_loss for s in batch_states])
    return (;
        max_violation = max_violation,
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
    new_max_violations = primal_state.max_violations
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
    trainer.prev_dual_training_state = deepcopy(trainer.dual_training_state)

    return
end

"""
    _stopping_criterion(iter::Int, method::AbstractL2OMethod, trainer::AbstractL2OTrainer)

Default stopping criterion for the L2O methods.
This function checks if the number of iterations has reached a predefined limit (100).
"""
function _stopping_criterion(iter::Int, ::M, ::N) where {M<:AbstractL2OMethod, N<:AbstractL2OTrainer}
    return iter >= 100 ? true : false
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
    method::AbstractPrimalDualMethod,
    trainer::AbstractPrimalDualTrainer,
    data,
)
    iter_primal = 1
    iter_dual = 1
    num_batches = length(data)
    current_state_primal = (;)
    current_state_dual = (;)
    _primal_loss = primal_loss(method)
    _dual_loss = dual_loss(method)
    train_state_primal = trainer.primal_training_state
    train_state_dual = trainer.dual_training_state

    # primal loop
    while primal_stopping_criterion(iter_primal, method, trainer)
        current_states_primal = Vector{NamedTuple}(undef, num_batches)
        iter_batch = 1
        for (θ) in data
            _, loss_val, stats, train_state_primal = Training.single_train_step!(
                AutoZygote(),          # AD backend
                _primal_loss,  # Loss function
                (θ, trainer), # Data
                train_state_primal, # Training state
            )
            current_states_primal[iter_batch] = stats
            iter_batch += 1
        end
        current_state_primal = reconcile_primal(trainer, current_states_primal)
        iter_primal += 1
    end

    # dual loop
    while dual_stopping_criterion(iter_dual, method, trainer)
        current_states_dual = Vector{NamedTuple}(undef, num_batches)
        iter_batch = 1
        for (θ) in data
            _, loss_val, stats, train_state_dual = Training.single_train_step!(
                AutoZygote(),          # AD backend
                _dual_loss,  # Loss function
                (θ, trainer), # Data
                train_state_dual, # Training state
            )
            current_states_dual[iter_batch] = stats
            iter_batch += 1
        end
        current_state_dual = reconcile_dual(trainer, current_states_dual)
        iter_dual += 1
    end
    # Update trainer state
    update_trainer!(method, trainer, current_state_primal, current_state_dual)
    return
end

end
