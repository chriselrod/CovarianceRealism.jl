module CovarianceRealism

using   SIMDPirates, SLEEFPirates, SLEEFwrap,
        LoopVectorization,
        StaticArrays,# BandedMatrices,
        Random,
        Distributions,
        Base.Cartesian,
        SpecialFunctions,
        UnsafeArrays,
        Statistics,
        LinearAlgebra,
        ScatteredArrays,
        VectorizationBase,
        KernelDensityDistributionEsimates, KernelDensity,
        RandomNumbers, VectorizedRNG,
        DifferentiableObjects,
        ForwardDiff,
        PaddedMatrices,
        Base.Threads
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
        pc2dfoster_RIC,
        InvCholCovar,
        sample_Pc!

using PaddedMatrices: MutableFixedSizeVector,
    MutableFixedSizeMatrix

# RandomNumbers.jl exports an AbstractRNG{T} that is a subset of AbstractRNG.
# We want our AbstractRNGs to refer to the more general Random.AbstractRNG.
import Random: AbstractRNG

set_zero_subnormals(true)

struct PCG_Scalar_and_Vector <: AbstractRNG
    scalar::RandomNumbers.PCG.PCGStateUnique{UInt64,Val{:RXS_M_XS},UInt64}
    vector::VectorizedRNG.PCG{2}
end
@inline Random.rand(pcg::PCG_Scalar_and_Vector) = rand(pcg.scalar)
@inline function Random.rand(pcg::PCG_Scalar_and_Vector, T::Union{Float16,Float32,Float64})
    rand(pcg.scalar, T)
end
@inline function Random.rand(pcg::PCG_Scalar_and_Vector, A::AbstractArray)
    rand(pcg.scalar, A)
end
@inline function Random.randn(pcg::PCG_Scalar_and_Vector)
    randn(pcg.scalar)
end
@inline function Random.randn(pcg::PCG_Scalar_and_Vector, T::Union{Float16,Float32,Float64})
    randn(pcg.scalar, T)
end
@inline function Random.randexp(pcg::PCG_Scalar_and_Vector)
    randexp(pcg.scalar)
end
@inline function Random.randexp(pcg::PCG_Scalar_and_Vector, T::Union{Float16,Float32,Float64})
    randexp(pcg.scalar, T)
end
@inline function Random.rand(pcg::PCG_Scalar_and_Vector, ::Type{NTuple{N,Core.VecElement{T}}}) where {N,T}
    rand(pcg.vector, NTuple{N,Core.VecElement{T}})
end
@inline function Random.randn(pcg::PCG_Scalar_and_Vector, ::Type{NTuple{N,Core.VecElement{T}}}) where {N,T}
    randn(pcg.vector, NTuple{N,Core.VecElement{T}})
end
@inline function Random.randexp(pcg::PCG_Scalar_and_Vector, ::Type{NTuple{N,Core.VecElement{T}}}) where {N,T}
    randexp(pcg.vector, NTuple{N,Core.VecElement{T}})
end
@inline function Random.rand(pcg::PCG_Scalar_and_Vector, ::Type{VectorizationBase.SVec{N,T}}) where {N,T}
    SVec(rand(pcg.vector, NTuple{N,Core.VecElement{T}}))
end
@inline function Random.randn(pcg::PCG_Scalar_and_Vector, ::Type{VectorizationBase.SVec{N,T}}) where {N,T}
    SVec(randn(pcg.vector, NTuple{N,Core.VecElement{T}}))
end
@inline function Random.randexp(pcg::PCG_Scalar_and_Vector, ::Type{VectorizationBase.SVec{N,T}}) where {N,T}
    SVec(randexp(pcg.vector, NTuple{N,Core.VecElement{T}}))
end

const PCG_Scalar = Union{PCG_Scalar_and_Vector, RandomNumbers.PCG.PCGStateUnique}
const PCG_Vector = Union{PCG_Scalar_and_Vector, VectorizedRNG.PCG}

const GLOBAL_PCG = PCG_Scalar_and_Vector(
    RandomNumbers.PCG.PCGStateUnique(PCG.PCG_RXS_M_XS),
    VectorizedRNG.PCG{2}()
)

include("utilities.jl")
include("distance_samples.jl")
include("ar_process.jl")
# include("kernel_density_estimate.jl")
# include("gaussian_process.jl")

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

# include("gaussian_process.jl")
include("RIC2ECI.jl")
include("pc_2d_foster.jl")

end # module
