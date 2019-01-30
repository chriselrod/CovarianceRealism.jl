
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


function silverman(data, α = 0.9)
    N = length(data)
    N <= 1 && return α
    var_width = std(data)
    q25, q75 = quantile(data, (0.25, 0.75))
    quant_width = (q75 - q25) * 0.7462686567164178

    width = min(var_width, quant_width)
    # @show width
    α * width * N^(-0.2)
end

function KernelDensity.kde(distances::UniformSamples)
    bw = silverman(distances.distances)
    if bw > 0
        return KernelDensity.kde( distances.distances, bandwidth = bw, npoints=2048 )
    else
        x = distances.distances[1]
        return UnivariateKDE(x:1.0:x,[1.0])
    end
end
function KernelDensity.kde(distances::WeightedSamples)
    bw = silverman(distances.distances)
    if bw > 0
        return KernelDensity.kde( distances.distances, bandwidth = bw, weights = distances.weights, npoints=2048 )
    else
        x = distances.distances[1]
        return UnivariateKDE(x:1.0:x,[1.0])
    end
end
