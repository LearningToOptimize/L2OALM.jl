# Contributing to L2OALM.jl

Thanks for helping! Issues and pull requests are welcome.

## Development setup

```sh
git clone https://github.com/LearningToOptimize/L2OALM.jl
cd L2OALM.jl
julia --project=test -e 'using Pkg; Pkg.instantiate()'   # Julia ≥ 1.11 reads [sources]
JULIA_NUM_THREADS=2 julia --project=test test/runtests.jl
julia --project=docs docs/make.jl                         # documentation
```

On Julia 1.10, which ignores `[sources]`, build the environment as CI does:

```sh
julia .github/ci/setup_test_env.jl test_env
julia --project=test_env test/runtests.jl
```

## Guidelines

  - **Theory first.** A new method or loss term cites its paper; the docs
    Theory page states the optimization problem with the paper's equation
    numbers, and code comments point to them ("eq. (n)").
  - **Test the mathematics.** A formula or property the code relies on gets
    a numerical test (identities against the paper's equations, gradients
    against finite differences, comparisons with a solver).
  - **Batch-first, array code.** The batch is the last dimension; losses use
    broadcasts and reductions so they differentiate and run on GPU arrays.
  - **Keep the core light.** ML frameworks, solvers and kernel backends are
    weak dependencies with package extensions.
  - **Format** with JuliaFormatter v1 (`using JuliaFormatter; format(".")`);
    CI checks it.
  - Small, focused commits; describe design choices in the pull request.
