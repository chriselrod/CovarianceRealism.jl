struct SquaredMahalanobisDistances{T,V <: AbstractVector{T}} <: AbstractVector{T}
    mahal::V
end
@inline Base.size(m::SquaredMahalanobisDistances) = size(m.mahal)
@inline Base.length(m::SquaredMahalanobisDistances) = length(m.mahal)
@inline Base.getindex(m::SquaredMahalanobisDistances, i) = m.mahal[i]
@inline Base.setindex!(m::SquaredMahalanobisDistances, v, i) = m.mahal[i] = v
@inline Base.resize!(m::SquaredMahalanobisDistances, N) = Base.resize!(m.mahal, N)
@inline Base.sort!(m::SquaredMahalanobisDistances) = sort!(m.mahal)
Base.IndexStyle(::Type{<:SquaredMahalanobisDistances}) = IndexLinear()

function SquaredMahalanobisDistances(Z::AbstractMatrix{T}) where T
    m = Vector{T}(undef, size(X,1))
    @inbounds @fastmath for i ∈ eachindex(m)
        m[i] = Z[i,1]^2 + Z[i,2]^2 + Z[i,3]^2
    end
    SquaredMahalanobisDistances(m)
end
function SquaredMahalanobisDistances!(m::SquaredMahalanobisDistances{T}, Z::AbstractMatrix{T}) where T
    resize!(m, size(Z,1))
    @inbounds @fastmath for i ∈ eachindex(m)
        m[i] = Z[i,1]^2 + Z[i,2]^2 + Z[i,3]^2
    end
    m
end
