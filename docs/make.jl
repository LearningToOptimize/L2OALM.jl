using Documenter
using DocumenterMermaid
using Literate
using L2OALM
using L2OCore
# Load the weak dependencies so that the extension docstrings are available.
using Lux
using BatchNLPKernels

const EXAMPLES = joinpath(@__DIR__, "src", "examples")
for file in readdir(EXAMPLES)
    endswith(file, ".jl") || continue
    Literate.markdown(joinpath(EXAMPLES, file), EXAMPLES; documenter = true, credit = false)
end

DocMeta.setdocmeta!(L2OALM, :DocTestSetup, :(using L2OALM, L2OCore); recursive = true)

makedocs(;
    modules = [
        L2OALM,
        Base.get_extension(L2OALM, :L2OALMLuxExt),
        Base.get_extension(L2OALM, :L2OALMBatchNLPKernelsExt),
    ],
    authors = "Andrew <arosemberg3@gatech.edu> and contributors",
    sitename = "L2OALM.jl",
    doctest = true,
    checkdocs = :exports,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://LearningToOptimize.github.io/L2OALM.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Package guide" => "guide.md",
        "Design" => "design.md",
        "Examples" => [
            "PDL on a parametric QP" => "examples/qp.md",
            "ExaModels and GPU kernels" => "examples/examodels.md",
        ],
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/LearningToOptimize/L2OALM.jl",
    devbranch = "main",
    push_preview = true,
)
