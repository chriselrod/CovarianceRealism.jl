
abstract type AbstractSamples{T} <: AbstractVector{T} end

struct WeightedSamples{T} <: AbstractSamples{T}
    distances::Vector{T}
    weights::Vector{T}
end

struct UniformSamples{T} <: AbstractSamples{T}
    distances::Vector{T}
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
