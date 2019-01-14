module CovarianceRealism

using   SIMDPirates, SLEEF, LoopVectorization,
        StaticOptim, StaticArrays,
        Random,
        Distributions,
        Base.Cartesian,
        SpecialFunctions,
        UnsafeArrays,
        Base.Threads, KissThreading,
        LinearAlgebra,
        ScatteredArrays,
        VectorizationBase,
        KernelDensityDistributionEsimates,
        RandomNumbers, VectorizedRNG
        # Gadfly # plotting

export  process_BPP!,
        run_sample!,
        thread_sample!,
        WorkDataChains,
        MCMCResult, MCMCResults,
        WeightedSamples,
        UniformSamples,
        ResizableMatrix,
        InverseWishart, CholInvWishart, RevCholWishart,
        MahalanobisDistances, PercentileMatch,
        sample_distances!,
        KDE,
        PercentileMatch!,
        mixture_fit,
        pc2dfoster_RIC

# RandomNumbers.jl exports an AbstractRNG{T} that is a subset of AbstractRNG.
# We want our AbstractRNGs to refer to the more general Random.AbstractRNG.
import Random: AbstractRNG

set_zero_subnormals(true)

struct PCG_Scalar_and_Vector
    scalar::RandomNumbers.PCG.PCGStateUnique{UInt64,Val{:RXS_M_XS},UInt64}
    vector::VectorizedRNG.PCG{2}
end

const PCG_Scalar = Union{PCG_Scalar_and_Vector, RandomNumbers.PCG.PCGStateUnique}
const PCG_Vector = Union{PCG_Scalar_and_Vector, VectorizedRNG.PCG}

const GLOBAL_PCG = PCG_Scalar_and_Vector(
    RandomNumbers.PCG.PCGStateUnique(PCG.PCG_RXS_M_XS),
    VectorizedRNG.PCG{2}()
)

include("utilities.jl")
include("distance_samples.jl")
# include("kernel_density_estimate.jl")

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
