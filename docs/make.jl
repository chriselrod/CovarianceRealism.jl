using Documenter, CovarianceRealism

makedocs(;
    modules=[CovarianceRealism],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/CovarianceRealism.jl/blob/{commit}{path}#L{line}",
    sitename="CovarianceRealism.jl",
    authors="Chris Elrod",
    assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/CovarianceRealism.jl",
    target="build",
    julia="1.0",
    deps=nothing,
    make=nothing,
)
