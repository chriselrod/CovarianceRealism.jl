module CovarianceRealism

using   SIMDPirates, SLEEFwrap,
        StaticOptim, StaticArrays,
        Random,
        Base.Cartesian,
        SpecialFunctions,
        UnsafeArrays,
        Base.Threads, KissThreading,
        LinearAlgebra,
        Distributions,
        KernelDensity,
        Interpolations,
        StatsBase, Statistics,
        Gadfly # plotting


export  process_BPP!,
        run_sample!,
        thread_sample!

include("utilities.jl")
include("distance_samples.jl")
include("kernel_density_estimate.jl")

include("mahalanobis_distances.jl")
include("misc_linear_algebra.jl") # old code, needs updating.

include("wisharts.jl")
include("groups.jl")
include("rng.jl")

include("mcmc_result.jl")
include("working_data.jl")

include("process_inputs.jl")
include("gibbs.jl")

include("simplex.jl")
include("mixture.jl")
include("percentile_matching.jl")

include("RIC2ECI.jl")
include("pc_2d_foster.jl")

end # module
