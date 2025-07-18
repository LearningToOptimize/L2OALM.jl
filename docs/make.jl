using L2OALM
using Documenter

DocMeta.setdocmeta!(L2OALM, :DocTestSetup, :(using L2OALM); recursive=true)

makedocs(;
    modules=[L2OALM],
    authors="Andrew <arosemberg3@gatech.edu> and contributors",
    sitename="L2OALM.jl",
    format=Documenter.HTML(;
        canonical="https://andrewrosemberg.github.io/L2OALM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/andrewrosemberg/L2OALM.jl",
    devbranch="main",
)
