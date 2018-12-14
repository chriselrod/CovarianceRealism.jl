
@noinline throwboundserror() = throw(BoundsError())

struct ResizableMatrix{T,NR} <: AbstractMatrix{T}
    data::Vector{T}
    nrows::Base.RefValue{Int}
end
@inline Base.size(rm::ResizableMatrix{T,NR}) where {T,NR} = (rm.nrows[], NR)
@inline Base.length(rm::ResizableMatrix) = length(rm.data)
@inline Base.getindex(rm::ResizableMatrix, i) = rm.data[i]
@inline Base.getindex(rm::ResizableMatrix, i, j) = rm.data[ i + (j-1)*rm.nrows[] ]
@inline Base.setindex!(rm::ResizableMatrix, v, i) = rm.data[i] = v
@inline Base.setindex!(rm::ResizableMatrix, v, i, j) = rm.data[i, j] = v
@inline Base.pointer(rm::ResizableMatrix) = pointer(rm.data)
Base.resize!(rm::ResizableMatrix{T,NR}, N) where {T,NR} = (rm.nrows[] = N; resize!(rm.data, NR*N))
Base.IndexStyle(::Type{<:ResizableMatrix}) = IndexLinear()
function ResizableMatrix{T,NR}(::UndefInitializer, N) where {T,NG}
    data = Vector{T}(undef, N * NG)
    ResizableMatrix{T,NR}(data, Ref(N))
end
