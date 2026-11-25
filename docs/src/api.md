```@meta
CurrentModule = L2OALM
```

# API reference

```@docs
L2OALM
```

## Method

```@docs
ALMMethod
AbstractAugmentation
PDLAugmentation
PHRAugmentation
AbstractBoundMap
NoBoundMap
SigmoidBoundMap
```

## Formulation and decomposition

```@docs
ALMFormulation
ALMDecomposition
num_rows
map_primal
project_multipliers
to_jump_duals
from_jump_duals
```

## Losses and penalty schedule

```@docs
primal_loss
augmentation_term
dual_target
dual_loss
violation_measure
update_penalty
```

## Guarantees

```@docs
DualConeFeasible
```

## Training

```@docs
ALMTrainer
train!
```

## L2OCore methods

```@docs
L2OCore.complete(::ALMMethod, ::L2OCore.L2OPipeline, ::Any, ::Any)
L2OCore.loss(::ALMMethod, ::L2OCore.L2OPipeline, ::Any, ::Any)
L2OCore.generate_data(::ALMMethod, ::L2OCore.L2OPipeline)
L2OCore.train_proxy!(::ALMMethod, ::L2OCore.L2OPipeline, ::ALMTrainer, ::Any)
```

## Extensions

```@autodocs
Modules = [Base.get_extension(L2OALM, :L2OALMLuxExt), Base.get_extension(L2OALM, :L2OALMBatchNLPKernelsExt)]
```

## Index

```@index
```
