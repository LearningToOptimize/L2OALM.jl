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
proxies (keywords `outer_iterations`, `primal_epochs`, `dual_epochs`,
`batchsize`, `rng`, `ad`, `callback`; see the package guide).

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
