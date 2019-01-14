
abstract type AbstractSamples{T} <: AbstractVector{T} end

struct WeightedSamples{T} <: AbstractSamples{T}
    distances::Vector{T}
    weights::Vector{T}
end

struct UniformSamples{T} <: AbstractSamples{T}
    distances::Vector{T}
end

function WeightedSamples{T}(::UndefInitializer, N) where T
    WeightedSamples(Vector{T}(undef, N),Vector{T}(undef, N))
end
function WeightedSamples{T}(::UndefInitializer, N, nthreads) where T
    samples = Vector{WeightedSamples{T}}(undef, nthreads)
    Threads.@threads for thread ∈ 1:nthreads
        samples[thread] = WeightedSamples{T}(undef, N)
    end
    samples
end
function UniformSamples{T}(::UndefInitializer, N) where T
    UniformSamples(Vector{T}(undef, N))
end
function UniformSamples{T}(::UndefInitializer, N, nthreads) where T
    samples = Vector{UniformSamples{T}}(undef, nthreads)
    Threads.@threads for thread ∈ 1:nthreads
        samples[thread] = UniformSamples{T}(undef, N)
    end
    samples
end

Base.size(s::AbstractSamples) = size(s.distances)
Base.length(s::AbstractSamples) = length(s.distances)
Base.getindex(ws::WeightedSamples, i) = (ws.distances[i], ws.weights[i])
Base.getindex(us::UniformSamples, i) = us.distances[i]
function Base.setindex!(ws::WeightedSamples, (d,w), i)
    ws.distances[i] = d
    ws.weights[i] = w
    (d,w)
end
Base.setindex!(us::UniformSamples, d, i) = us.distances[i] = d
Base.IndexStyle(::Type{<:AbstractSamples}) = IndexLinear()
function Base.resize!(s::WeightedSamples, N)
    resize!(s.distances, N)
    resize!(s.weights)
    s
end
function Base.resize!(s::UniformSamples, N)
    resize!(s.distances, N)
    s
end

function KernelDensityDistributionEsimates.KDE(distances::UniformSamples)
    KDE(KernelDensityDistributionEsimates.kde(distances.distances,npoints=2048))
end
function KernelDensityDistributionEsimates.KDE(distances::WeightedSamples)
    KDE(KernelDensityDistributionEsimates.kde(distances.distances, weights = distances.weights,npoints=2048))
end
