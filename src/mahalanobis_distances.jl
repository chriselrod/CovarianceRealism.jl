struct MahalanobisDistances{T} <: AbstractVector{T}
    mahal::Vector{T}
end
@inline Base.size(m::MahalanobisDistances) = size(m.mahal)
@inline Base.length(m::MahalanobisDistances) = length(m.mahal)
@inline Base.getindex(m::MahalanobisDistances, i) = m.mahal[i]
@inline function Base.getindex(m::S, I::AbstractArray) where {S<:MahalanobisDistances}
    MahalanobisDistances(m.mahal[I])
end
@inline Base.setindex!(m::MahalanobisDistances, v, i) = m.mahal[i] = v
@inline Base.resize!(m::MahalanobisDistances, N) = Base.resize!(m.mahal, N)
@inline Base.sort!(m::MahalanobisDistances) = sort!(m.mahal)
Base.IndexStyle(::Type{<:MahalanobisDistances}) = IndexLinear()

MahalanobisDistances{T}(undef, N) where T = MahalanobisDistances(Vector{T}(undef, N))
function MahalanobisDistances(Z::AbstractMatrix{T}) where T
    m = Vector{T}(undef, size(X,1))
    @inbounds @fastmath for i ∈ eachindex(m)
        m[i] = sqrt(Z[i,1]^2 + Z[i,2]^2 + Z[i,3]^2)
    end
    MahalanobisDistances(m)
end
function MahalanobisDistances!(m::MahalanobisDistances{T}, Z::AbstractMatrix{T}) where T
    resize!(m, size(Z,1))
    @inbounds @fastmath for i ∈ eachindex(m)
        m[i] = sqrt(Z[i,1]^2 + Z[i,2]^2 + Z[i,3]^2)
    end
    m
end
