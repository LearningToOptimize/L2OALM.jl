# # PDL on a parametric QP
#
# This example trains primal and dual proxies with PDL (Algorithm 1 of Park &
# Van Hentenryck, 2023) for a small convex QP whose right-hand sides are
# parameters, and compares the predictions with Ipopt.

using JuMP, Ipopt, L2OCore, L2OALM
using Lux, Optimisers, Zygote, Random

# ## The problem
#
# ``\min_x \|x\|^2 + x_1 - 3x_3`` subject to ``x_1 + x_2 + x_3 = p_1``,
# ``x_1 - x_2 \le p_2``, ``x_2 + 2x_3 \ge -1`` and ``-2 \le x \le 2``.

model = Model()
@variable(model, p[1:2] in MOI.Parameter(1.0))
@variable(model, -2 <= x[1:3] <= 2)
@constraint(model, x[1] + x[2] + x[3] == p[1])
@constraint(model, x[1] - x[2] <= p[2])
@constraint(model, x[2] + 2x[3] >= -1)
@objective(model, Min, sum(x .^ 2) + x[1] - 3x[3])
problem = ParametricProblem(model)

# ## Method and pipeline
#
# The pipeline checks that the problem class is supported and records the
# row bounds; the formulation is what the losses see.

method = ALMMethod(; ρ_init = 1.0, α = 2.0, τ = 0.8, ρmax = 1e3)
pipe = L2OPipeline(problem, method)
form = ALMFormulation(pipe)

# ## Self-supervised data
#
# Only parameters are sampled; no solver is called.

rng = Random.Xoshiro(42)
sampler = BoxSampler(fill(-1.0, 2), fill(1.0, 2))
train_data = generate_data(pipe; sampler = sampler, n = 256, rng = rng)
num_samples(train_data)

# ## Proxies
#
# The dual proxy outputs one multiplier per constraint row. Its last layer
# starts at zero, so the first primal phase is the plain penalty method.

nλ = num_rows(form)
primal = Chain(Dense(2, 32, tanh), Dense(32, 32, tanh), Dense(32, 3))
dual = Chain(Dense(2, 32, tanh), Dense(32, nλ; init_weight = zeros64, init_bias = zeros64))
ps, st = Lux.setup(rng, primal)
pd, sd = Lux.setup(rng, dual)
trainer = ALMTrainer(
    Training.TrainState(primal, f64(ps), st, Optimisers.Adam(1e-2)),
    Training.TrainState(dual, f64(pd), sd, Optimisers.Adam(1e-2));
    ρ = method.ρ_init,
)

# ## Training (Algorithm 1)

L2OALM.train!(
    trainer,
    pipe,
    train_data;
    outer_iterations = 10,
    primal_epochs = 15,
    dual_epochs = 15,
    batchsize = 64,
    rng = rng,
)
[(h.iteration, h.ρ, round(h.violation; sigdigits = 3)) for h in trainer.history]

# The penalty grows by `α` whenever the violation measure of eq. (8) did not
# drop by the factor `τ`.

# ## Evaluation against Ipopt

P = L2OCore.sample_parameters(sampler, rng, 32)
pred = complete(pipe, P, L2OCore.predict_batch(trainer, P))
report = evaluate(pipe, P, pred)
reference = solve_instances(problem, P; optimizer = Ipopt.Optimizer)
gap = abs.(report.objective .- reference.objective) ./ max.(1, abs.(reference.objective))
(
    mean_gap = sum(gap) / length(gap),
    mean_max_violation = sum(report.max_violation) / length(gap),
)
