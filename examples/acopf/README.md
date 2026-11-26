# AC-OPF with PDL (smoke example)

This is a smoke example: it runs the whole pipeline on AC-OPF but is not a
benchmark and does not reproduce the AC-OPF results of the paper.

`train.jl` trains PDL proxies for a parametric AC optimal power flow (load
multipliers at every bus as parameters) built in ExaModels by `power.jl`, and
evaluates the batch with BatchNLPKernels kernels.

```sh
julia --project=examples/acopf -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/acopf examples/acopf/train.jl pglib_opf_case14_ieee.m 10
```

For an NVIDIA GPU, `]add CUDA LuxCUDA` in this environment and run with
`L2OALM_CUDA=1`. The script uses `SigmoidBoundMap` (variable bounds hold by
construction) and `PHRAugmentation`.

The hyper-parameters are not tuned: with the defaults above a few outer
iterations exercise the full pipeline (kernels, losses, Algorithm 1), and the
constraint violations stay large. Reaching the accuracy of the paper needs many more inner steps per phase
(thousands rather than hundreds), a warm-up at a larger `ρ`, and problem-
specific output layers; see the design page of the documentation.
