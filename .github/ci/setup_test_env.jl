# Build a test environment that works on every supported Julia version.
#
# Julia 1.10 ignores the [sources] section of test/Project.toml, so the
# unregistered packages (L2OALM itself, L2OCore, BatchNLPKernels) are removed
# from a copy of it and added by path/URL instead. Usage:
#
#     julia .github/ci/setup_test_env.jl <env_dir>
#     julia --project=<env_dir> test/runtests.jl
using Pkg, TOML

const ROOT = dirname(dirname(@__DIR__))
const UNREGISTERED = [
    PackageSpec(
        url = "https://github.com/LearningToOptimize/LearningToOptimize.jl",
        subdir = "lib/L2OCore",
        rev = "ar/l2ocore-design",
    ),
    PackageSpec(url = "https://github.com/LearningToOptimize/BatchNLPKernels.jl", rev = "main"),
]

env = abspath(get(ARGS, 1, joinpath(ROOT, "test_env")))
mkpath(env)
project = TOML.parsefile(joinpath(ROOT, "test", "Project.toml"))
for name in ("L2OALM", "L2OCore", "BatchNLPKernels")
    delete!(project["deps"], name)
    haskey(project, "compat") && delete!(project["compat"], name)
end
delete!(project, "sources")
open(io -> TOML.print(io, project), joinpath(env, "Project.toml"), "w")
rm(joinpath(env, "Manifest.toml"); force = true)

Pkg.activate(env)
Pkg.add(UNREGISTERED)
Pkg.develop(path = ROOT)
Pkg.instantiate()
