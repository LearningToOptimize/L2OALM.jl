"""
    L2OALM

Self-supervised primal-dual learning with an augmented Lagrangian (PDL,
Park & Van Hentenryck, AAAI 2023, arXiv:2208.09046) as an
[`L2OCore`](https://github.com/LearningToOptimize/LearningToOptimize.jl/tree/main/lib/L2OCore)
primal-dual method.

The core of the package is framework-agnostic array code: the primal loss
(eq. (5) of the paper), the dual targets (eq. (6)), the dual loss (eq. (7)),
the violation measure (eq. (8)) and the penalty rule (eq. (9)) are plain
functions of batches of parameters, primal points and multipliers, written
with broadcasts and reductions only so that they differentiate with array AD
and run on GPU arrays. Package extensions add

  - `L2OALMLuxExt` (load `Lux`): the training loop of Algorithm 1 for two
    `Lux.Training.TrainState`s (`L2OALM.train!`, also reachable as
    `L2OCore.train_proxy!`);
  - `L2OALMBatchNLPKernelsExt` (load `BatchNLPKernels`): an evaluator for
    ExaModels problems that evaluates the batch with KernelAbstractions
    kernels (CPU or GPU).
"""
module L2OALM

import ChainRulesCore
import JuMP
import L2OCore
import Random

using L2OCore:
    AbstractEvaluator,
    AbstractPrimalDualMethod,
    L2ODataset,
    L2OPipeline,
    ParametricProblem,
    Prediction

include("formulation.jl")
include("method.jl")
include("losses.jl")
include("evaluation.jl")
include("trainer.jl")

export ALMMethod,
    AbstractAugmentation,
    PDLAugmentation,
    PHRAugmentation,
    AbstractBoundMap,
    NoBoundMap,
    SigmoidBoundMap,
    HardSigmoidBoundMap,
    enforces_bounds,
    ALMFormulation,
    ALMDecomposition,
    num_rows,
    map_primal,
    project_multipliers,
    to_jump_duals,
    from_jump_duals,
    primal_loss,
    augmentation_term,
    dual_target,
    dual_loss,
    violation_measure,
    instance_violations,
    violation_report,
    row_penalty,
    update_penalty,
    dual_sign_violation,
    ALMTrainer

end # module L2OALM
