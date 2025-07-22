module L2OALM

using BatchNLPKernels
using ExaModels

using Lux
using LuxCUDA
using Lux.Training
using MLUtils
using Optimisers
using CUDA

# using OpenCL, pocl_jll, AcceleratedKernels

"""
    LagrangianDualLoss(;max_dual=1e6)

Returns a function that computes the MSE loss for the (dual-)model predicting lagrangian dual variables
from constraint evaluations `gh` and the current dual predictions `dual_hat_k`.
Target is calculated using the augmented lagrangian method. 
Target dual variables are clipped from zero to `max_dual`.

Keywords:
    - `max_dual`: Maximum value for the target dual variables.
"""
function LagrangianDualLoss(;max_dual=1e6)
    return (dual_model, ps_dual, st_dual, data) -> begin
        x, hpm, dual_hat_k, gh = data
        ρ = hpm.ρ
        # Get current dual predictions
        dual_hat, st_dual_new = dual_model(x, ps_dual, st_dual)
        
        # Separate bound and equality constraints
        gh_bound = gh[1:end-n_bus*2,:]
        gh_equal = gh[end-n_bus*2+1:end,:]
        dual_hat_bound = dual_hat_k[1:end-n_bus*2,:]
        dual_hat_equal = dual_hat_k[end-n_bus*2+1:end,:]
        
        # Target for dual variables
        dual_target = vcat(
            min.(max.(dual_hat_bound + ρ .* gh_bound, 0), max_dual),
            min.(dual_hat_equal + ρ .* gh_equal, max_dual)
        )
        
        loss = mean((dual_hat .- dual_target).^2)
        return loss, st_dual_new, (dual_loss=loss,)
    end
end

"""
    LagrangianPrimalLoss(bm::BatchModel)

Returns a function that computes the augmented lagrangian primal loss
from current dual predictions `dual_hat` for the batch model `bm` under parameters `Θ`.

Arguments:
    - `bm`: A `BatchModel` instance that contains the model and batch configuration.
"""
function LagrangianPrimalLoss(bm::BatchModel)    
    return (model, ps, st, data) -> begin
        Θ, hpm, dual_hat = data
        ρ = hpm.ρ
        num_s = size(Θ, 2)

        # Forward pass for prediction
        X̂, st_new = model(Θ, ps, st)
        
        # Calculate violations and objectives
        objs = BNK.objective!(bm, X̂, Θ)
        # gh = constraints!(bm, X̂, Θ)
        Vc, Vb = BNK.all_violations!(bm, X̂, Θ)
        V = vcat(Vb, Vc)
        total_loss = (
            sum(abs.(dual_hat .* V)) / num_s +
            ρ / 2 * sum((V).^2) / num_s +
            mean(objs)
        )

        return total_loss, st_new, (
            total_loss=total_loss,
            mean_violations=mean(V),
            new_max_violation=maximum(V),
            mean_objs=mean(objs),
        )
    end
end

"""
    TrainingStepLoop

A structure to define a training step loop for the L2O-ALM algorithm.

Fields:
- `loss_fn`: Function to compute the loss for the training step.
- `stopping_criteria`: Vector of functions that determine when to stop the training loop.
- `hyperparameters`: Dictionary of hyperparameters and hyper-states for the training step.
- `parameter_update_fns`: Vector of functions to update hyperparameters after each training step.
- `reconcile_state`: Function to reconcile the state after processing a batch of data.
"""
mutable struct TrainingStepLoop
    loss_fn::Function
    stopping_criteria::Vector{Function}
    hyperparameters::Dict{Symbol, Any}
    parameter_update_fns::Vector{Function}
    reconcile_state::Function
    pre_hook::Function
end

"""
    _pre_hook_primal(θ, primal_model, train_state_primal, dual_model, train_state_dual, bm)

Default pre-hook function for the primal model in the L2O-ALM algorithm.
This function performs a forward pass through the dual model to obtain the dual predictions.
"""
function _pre_hook_primal(θ, primal_model, train_state_primal, dual_model, train_state_dual, bm)
    # Forward pass for dual model
    dual_hat_k, _ = dual_model(θ, train_state_dual.parameters, train_state_dual.state)

    return (dual_hat_k,)
end

"""
    _pre_hook_dual(θ, primal_model, train_state_primal, dual_model, train_state_dual, bm)

Default pre-hook function for the dual model in the L2O-ALM algorithm.
This function performs a forward pass through the primal model to obtain the predicted state and constraints.
"""
function _pre_hook_dual(θ, primal_model, train_state_primal, dual_model, train_state_dual, bm)
    # # Forward pass for primal model
    X̂, _ = primal_model(θ, train_state_primal.parameters, train_state_primal.state)
    gh = constraints!(bm, X̂, Θ)
    
    # Forward pass for dual model
    dual_hat, _ = dual_model(θ, train_state_dual.parameters, train_state_dual.state)

    return (dual_hat, gh,)
end

"""
    _reconcile_alm_primal_state(batch_states::Vector{NamedTuple})

Default function that reconciles the state of the primal model after processing a batch of data.
This function computes the maximum violation, mean violations, mean objectives, and total loss
from the batch states.
"""
function _reconcile_alm_primal_state(batch_states::Vector{NamedTuple})
    max_violation = maximum([s.new_max_violation for s in batch_states])
    mean_violations = mean([s.mean_violations for s in batch_states])
    mean_objs = mean([s.mean_objs for s in batch_states])
    mean_loss = mean([s.total_loss for s in batch_states])
    return (;
        new_max_violation=max_violation,
        mean_violations=mean_violations,
        mean_objs=mean_objs,
        total_loss=mean_loss,
    )
end

"""
    _reconcile_alm_dual_state(batch_states::Vector{NamedTuple})

Default function that reconciles the state of the dual model after processing a batch of data.
This function computes the mean dual loss from the batch states.
"""
function _reconcile_alm_dual_state(batch_states::Vector{NamedTuple})
    dual_loss = mean([s.dual_loss for s in batch_states])
    return (dual_loss=dual_loss,)
end

"""
    _update_ALM_ρ!(hpm::Dict{Symbol, Any}, current_state::NamedTuple)

Default function to update the hyperparameter ρ in the ALM algorithm.
This function increases ρ by a factor of α if the new maximum violation exceeds τ times the previous maximum violation.
"""
function _update_ALM_ρ!(hpm::Dict{Symbol, Any}, current_state::NamedTuple)
    if current_state.new_max_violation > hpm.τ * hpm.max_violation
        hpm.ρ = min(hpm.ρmax, hpm.ρ * hpm.α)
    end
    hpm.max_violation = current_state.new_max_violation
    return
end

"""
    _default_primal_loop(bm::BatchModel)

Returns a default `TrainingStepLoop` for the primal model in the L2O-ALM algorithm.
"""
function _default_primal_loop(bm::BatchModel)
    return TrainingStepLoop(
        LagrangianPrimalLoss(bm),
        [(iter, current_state, hpm) -> iter >= 100 ? true : false],
        Dict{Symbol, Any}(
            :ρ => 1.0,
            :ρmax => 1e6,
            :τ => 0.8,
            :α => 10.0,
            max_violation => 0.0,
        ),
        [_update_ALM_ρ!],
        _reconcile_alm_primal_state,
        _pre_hook_primal
    )
end

"""
    _default_dual_loop()

Returns a default `TrainingStepLoop` for the dual model in the L2O-ALM algorithm.
"""
function _default_dual_loop()
    return TrainingStepLoop(
        LagrangianDualLoss(),
        [(iter, current_state, hpm) -> iter >= 100 ? true : false],
        Dict{Symbol, Any}(
            :max_dual => 1e6,
        ),
        [],
        _reconcile_alm_dual_state,
        _pre_hook_dual
    )
end

"""
    L2OALM_epoch(bm::BatchModel, primal_model::Lux.Model, train_state_primal::Lux.TrainingState,
                 dual_model::Lux.Model, train_state_dual::Lux.TrainingState,
                 training_step_loop_primal::TrainingStepLoop=_default_primal_loop(bm),
                 training_step_loop_dual::TrainingStepLoop=_default_dual_loop(),
                 data)

Runs a single epoch of the L2O-ALM algorithm, training both primal and dual models.

Arguments:
- `bm`: A `BatchModel` instance that contains the model and batch configuration.
- `primal_model`: The Lux model for the primal problem.
- `train_state_primal`: The training state for the primal model.
- `dual_model`: The Lux model for the dual problem.
- `train_state_dual`: The training state for the dual model.
- `training_step_loop_primal`: The training step loop for the primal model.
- `training_step_loop_dual`: The training step loop for the dual model.
- `data`: The training data, typically a collection of batches.
"""
function L2OALM_epoch(
    bm::BatchModel,
    primal_model::Lux.Model,
    train_state_primal::Lux.TrainingState,
    dual_model::Lux.Model,
    train_state_dual::Lux.TrainingState,
    training_step_loop_primal::TrainingStepLoop=_default_primal_loop(bm),
    training_step_loop_dual::TrainingStepLoop=_default_dual_loop(),
    data
)
    iter_primal = 1
    iter_dual = 1
    num_batches = length(data)
    current_state_primal = (;)
    current_state_dual = (;)

    # primal loop
    while all(stopping_criterion(iter_primal, current_state_primal, training_step_loop_primal.hyperparameters) for stopping_criterion in training_step_loop_primal.stopping_criteria)
        current_states_primal = Vector{NamedTuple}(undef, num_batches)
        iter_batch = 1
        for (θ) in data
            _, loss_val, stats, train_state_primal = Training.single_train_step!(
                AutoZygote(),          # AD backend
                training_step_loop_primal.loss_fn,  # Loss function
                (θ, training_step_loop_primal.hyperparameters, training_step_loop_primal.pre_hook(θ, primal_model, train_state_primal, dual_model, train_state_dual)...), # Data
                train_state_primal # Training state
            )
            current_states_primal[iter_batch] = stats
            iter_batch += 1
        end
        current_state_primal = training_step_loop_primal.reconcile_state(current_states_primal)
        iter_primal += 1
    end
    for fn in training_step_loop_primal.parameter_update_fns
        fn(training_step_loop_primal.hyperparameters, current_state_primal)
    end

    # dual loop
    while all(stopping_criterion(iter_dual, current_state_dual, training_step_loop_dual.hyperparameters) for stopping_criterion in training_step_loop_dual.stopping_criteria)
        current_states_dual = Vector{NamedTuple}(undef, num_batches)
        iter_batch = 1
        for (θ) in data
            _, loss_val, stats, train_state_dual = Training.single_train_step!(
                AutoZygote(),          # AD backend
                training_step_loop_dual.loss_fn,  # Loss function
                (θ, training_step_loop_dual.hyperparameters, training_step_loop_dual.pre_hook(θ, primal_model, train_state_primal, dual_model, train_state_dual)...), # Data
                train_state_dual # Training state
            )
            current_states_dual[iter_batch] = stats
            iter_batch += 1
        end
        current_state_dual = training_step_loop_dual.reconcile_state(current_states_dual)
        iter_dual += 1
    end
    for fn in training_step_loop_dual.parameter_update_fns
        fn(training_step_loop_dual.hyperparameters, current_state_dual)
    end
    return
end

"""
    L2OALM_train(bm::BatchModel, primal_model::Lux.Model, dual_model::Lux.Model,
        train_state_primal::Lux.TrainingState, train_state_dual::Lux.TrainingState,
        training_step_loop_primal::TrainingStepLoop=_default_primal_loop(bm),
        training_step_loop_dual::TrainingStepLoop=_default_dual_loop(),
        stopping_criteria::Vector{Function}=[(iter, primal_model::Lux.Model, dual_model::Lux.Model,
            train_state_primal::Lux.TrainingState, train_state_dual::Lux.TrainingState) -> iter >= 100 ? true : false],
        data
    )

Runs the L2O-ALM training algorithm until the stopping criteria are met.

Arguments:
- `bm`: A `BatchModel` instance that contains the model and batch configuration.
- `primal_model`: The Lux model for the primal problem.
- `dual_model`: The Lux model for the dual problem.
- `train_state_primal`: The training state for the primal model.
- `train_state_dual`: The training state for the dual model.
- `training_step_loop_primal`: The training step loop for the primal model.
- `training_step_loop_dual`: The training step loop for the dual model.
- `stopping_criteria`: A vector of functions that determine when to stop the training loop.
- `data`: The training data, typically a collection of batches.
"""
function L2OALM_train(
    bm::BatchModel,
    primal_model::Lux.Model,
    dual_model::Lux.Model,
    train_state_primal::Lux.TrainingState,
    train_state_dual::Lux.TrainingState,
    training_step_loop_primal::TrainingStepLoop=_default_primal_loop(bm),
    training_step_loop_dual::TrainingStepLoop=_default_dual_loop(),
    stopping_criteria::Vector{Function}=[(iter, current_state, hpm) -> iter >= 100 ? true : false],
    data
)
    iter = 1
    while all(stopping_criterion(iter, primal_model, dual_model, train_state_primal, train_state_dual) for stopping_criterion in stopping_criteria)
        L2OALM_epoch(
            bm,
            primal_model,
            train_state_primal,
            dual_model,
            train_state_dual,
            training_step_loop_primal,
            training_step_loop_dual,
            data
        )
        iter += 1
    end
    return
end

end
