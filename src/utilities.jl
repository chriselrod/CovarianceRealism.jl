
@noinline throwboundserror() = throw(BoundsError())

struct ResizableMatrix{T,NC} <: AbstractMatrix{T}
    data::Vector{T}
    nrows::Base.RefValue{Int}
end
@inline Base.size(rm::ResizableMatrix{T,NC}) where {T,NC} = (rm.nrows[], NC)
@inline Base.length(rm::ResizableMatrix) = length(rm.data)
@inline Base.getindex(rm::ResizableMatrix, i::Integer) = rm.data[i]
@inline Base.getindex(rm::ResizableMatrix, I::CartesianIndex{2}) = rm.data[I[1] + (I[2]-1)*rm.nrows[]]
@inline Base.getindex(rm::ResizableMatrix, i::Integer, j::Integer) = rm.data[ i + (j-1)*rm.nrows[] ]
@inline Base.setindex!(rm::ResizableMatrix, v, i) = rm.data[i] = v
@inline Base.setindex!(rm::ResizableMatrix, v, i::Integer, j::Integer) = rm.data[i + (j-1)*rm.nrows[]] = v
@inline Base.pointer(rm::ResizableMatrix) = pointer(rm.data)
@inline Base.pointer(rm::ResizableMatrix, i::Integer) = pointer(rm.data, i)
Base.resize!(rm::ResizableMatrix{T,NC}, N) where {T,NC} = (rm.nrows[] = N; resize!(rm.data, NC*N))
Base.IndexStyle(::Type{<:ResizableMatrix}) = IndexLinear()
function ResizableMatrix{T,NC}(::UndefInitializer, N) where {T,NC}
    data = Vector{T}(undef, N * NC)
    ResizableMatrix{T,NC}(data, Ref(N))
end

function Base.getindex(rm::ResizableMatrix{T,NC}, I::Union{<:AbstractArray{Bool},<:BitArray}, ::Colon) where {T,NC}
    @boundscheck rm.nrows[] < length(I) && throwboundserror()
    out = Matrix{T}(undef, sum(I), NC)
    nrows = rm.nrows[]
    @inbounds for c ∈ 1:NC
        out[:,c] .= @view(rm.data[1+(c-1)*nrows:c*nrows])[I]
    end
    out
end
@inline function Base.view(rm::ResizableMatrix{T,NC}, ::Colon, i::Integer) where {T,NC}
    @boundscheck i > NC && throwboundserror()
    nrows = rm.nrows[]
    @view rm.data[ (1:nrows) .+ (i-1)*nrows ]
end


@generated function findall_zerorows(X::AbstractMatrix{T}, ::Val{ncol}) where {T,ncol}
    @assert ncol > 1
    ex = :((X[i,$(ncol-1)] != zero($T)) | (X[i,$ncol] != zero($T)))
    for j ∈ ncol-2:-1:1
        ex = :( (X[i,$j] != zero($T)) | $ex )
    end
    q = quote
        tb_1 = $(Expr(:tuple, [:(X[iu+$(j-1),1] != zero($T)) for j ∈ 1:16]...))
    end
    for c ∈ 2:ncol
        push!(q.args, quote
            $(Symbol(:tb_, c)) = ($(Symbol(:tb_, c-1)) .| $(Expr(:tuple, [:(X[iu+$(j-1),$c] != zero($T)) for j ∈ 1:16]...)))
        end)
    end
    quote
        N = size(X,1)
        N16, r = divrem(N,16)
        z0 = Vector{Bool}(undef, N)
        @inbounds @fastmath for iu ∈ 1:16:N
            $q
            z0[iu:iu+15] .= $(Symbol(:tb_, ncol))
        end
        @inbounds @fastmath for i ∈ N-r+1:N
            z0[i] = $ex
        end
        z0
    end
end
