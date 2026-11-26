"""
    ALMTrainer(primal, dual; ρ = 1.0)

Mutable training state of PDL (Algorithm 1 of Park & Van Hentenryck, 2023):

  - `primal`, `dual`: framework-specific states of the primal proxy (outputs
    `num_decision_variables` rows) and of the dual proxy (outputs `num_rows`
    multipliers, L2OALM convention); with the Lux extension these are
    `Lux.Training.TrainState`s;
  - `ρ`: current penalty; `v_prev`: violation measure of the previous outer
    iteration (`Inf` before the first one); `iteration`: completed outer
    iterations;
  - `options`: free-form state of the training loop that must survive between
    calls (e.g. the current learning rates when `lr_decay` is used);
  - `history`: one `NamedTuple` per outer iteration with fields `iteration`,
    `ρ` (penalty used in that iteration), `violation` (eq. (8)), `primal_loss`
    and `dual_loss` (means over the last epoch of each phase).

Initialize the last layer of the dual proxy at zero so that the first primal
phase minimizes the plain penalty loss, as in the paper.
"""
mutable struct ALMTrainer{P,D,T<:Real}
    primal::P
    dual::D
    ρ::T
    v_prev::T
    iteration::Int
    history::Vector{
        NamedTuple{
            (:iteration, :ρ, :violation, :primal_loss, :dual_loss),
            Tuple{Int,T,T,T,T},
        },
    }
    options::Dict{Symbol,Any}
end

function ALMTrainer(primal, dual; ρ::Real = 1.0)
    T = float(typeof(ρ))
    return ALMTrainer(
        primal,
        dual,
        T(ρ),
        T(Inf),
        0,
        NamedTuple{
            (:iteration, :ρ, :violation, :primal_loss, :dual_loss),
            Tuple{Int,T,T,T,T},
        }[],
        Dict{Symbol,Any}(),
    )
end

"""
    train!(trainer::ALMTrainer, method::ALMMethod, form::ALMFormulation, inputs; kwargs...)
    train!(trainer::ALMTrainer, pipe::L2OPipeline, inputs; kwargs...)

Run outer iterations of PDL (Algorithm 1) on the parameter matrix `inputs`
(`num_parameters × N`, or an `L2OCore.L2ODataset`), updating `trainer` in
place. Each outer iteration `k`:

 1. **primal phase**: `primal_epochs` epochs minimizing [`primal_loss`](@ref)
    (eq. (5)) with the dual proxy frozen;
 2. computes `v_k` with [`violation_measure`](@ref) (eq. (8)) over `inputs`;
 3. freezes a copy of the dual proxy (`λ_k`);
 4. **dual phase**: `dual_epochs` epochs minimizing [`dual_loss`](@ref)
    (eq. (7)) towards [`dual_target`](@ref) (eq. (6)) with the primal proxy and
    `λ_k` frozen;
 5. updates `ρ` with [`update_penalty`](@ref) (eq. (9)).

Methods are provided by extensions: load `Lux` for `Lux.Training.TrainState`
proxies. Keywords of the Lux method:

  - `outer_iterations` (`K`), `primal_epochs`, `dual_epochs` (`L`, in epochs),
    `batchsize`, `rng`, `ad` (an ADTypes backend, default `AutoZygote()`),
    `callback(trainer, entry)` after each outer iteration;
  - `violation_reduction = maximum`: reduction over the per-instance violations
    ([`instance_violations`](@ref)) giving `v_k`; `maximum` is eq. (8), `mean`
    is less sensitive to a few hard instances;
  - `validation = nothing`, `lr_decay = 1`: with a validation parameter matrix
    and `lr_decay < 1`, after every epoch of a phase the phase loss is
    evaluated on `validation` and the learning rate of that phase is
    multiplied by `lr_decay` whenever the loss exceeds the best value of the
    phase so far (App. C.3 of the paper uses `0.99`). The current rates are
    kept in `trainer.options[:primal_eta]`/`[:dual_eta]` and the number of
    decays in `[:primal_decays]`/`[:dual_decays]`; the optimiser must have an
    `eta` field (Adam, Descent, …).

`train!` is not exported, because the name would clash with `Flux.train!`;
call it as `L2OALM.train!`, or through the generic
`L2OCore.train_proxy!(pipe.method, pipe, trainer, inputs; kwargs...)`.
"""
function train! end

train!(trainer::ALMTrainer, pipe::L2OPipeline, inputs; kwargs...) =
    train!(trainer, pipe.method, ALMFormulation(pipe), inputs; kwargs...)
train!(
    trainer::ALMTrainer,
    method::ALMMethod,
    form::ALMFormulation,
    data::L2ODataset;
    kwargs...,
) = train!(trainer, method, form, data.inputs; kwargs...)

"""
    L2OCore.train_proxy!(method::ALMMethod, pipe, trainer::ALMTrainer, data; kwargs...)

L2OCore entry point; forwards to [`train!`](@ref).
"""
L2OCore.train_proxy!(
    method::ALMMethod,
    pipe::L2OPipeline,
    trainer::ALMTrainer,
    data;
    kwargs...,
) = train!(trainer, method, ALMFormulation(pipe), data; kwargs...)
