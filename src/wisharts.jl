
abstract type Wishart3x3{T} <: AbstractMatrix{T} end

struct InverseWishart{T} <: Wishart3x3{T}
    data::NTuple{8,VE{T}}
end
struct CholInvWishart{T} <: Wishart3x3{T}
    data::NTuple{8,T}
end
struct RevCholWishart{T} <: Wishart3x3{T}
    data::NTuple{6,T}
end
@inline InverseWishart(x::Vararg{T,8}   ) where T = InverseWishart{T}(VE.(x))
@inline InverseWishart{T}(x::Vararg{T,8}) where T = InverseWishart{T}(VE.(x))
@inline CholInvWishart(x::Vararg{T,8}   ) where T = CholInvWishart{T}(x)
@inline CholInvWishart{T}(x::Vararg{T,8}) where T = CholInvWishart{T}(x)
@inline CholInvWishart{T}(x::Vararg{T,6}) where T = CholInvWishart{T}(x...,T(1),T(1))
@inline RevCholWishart(x::Vararg{T,6}   ) where T = RevCholWishart{T}(x)
@inline RevCholWishart{T}(x::Vararg{T,6}) where T = RevCholWishart{T}(x)
@inline CholInvWishart{T}(x::Vararg{SVec{W,T},8}) where {W,T} = CholInvWishart{SVec{W,T}}(x)
# @inline function SArray{Tuple{S},T,N,L}(vs::Vararg{SIMDPirates.SVec{W,T},L}) where {S,T,N,L,W}
#     SArray{S}(vs...)
# end
# @inline Base.size(::Wishart2x2) = (2,2)
# @inline Base.length(::Wishart2x2) = 4
@inline Base.size(::Wishart3x3) = (3,3)
@inline Base.length(::Wishart3x3) = 9
# ScatteredArrays.type_length(::Type{Wishart2x2}) = 3
# ScatteredArrays.type_length(::Type{<:Wishart3x3}) = 6
# ScatteredArrays.type_length(::Type{CholInvWishart{T}}) where T = 6

@inline extractval(x::Core.VecElement{T}) where T = x.value
@inline extractval(x) = x
@inline Base.getindex(w::Wishart3x3, i::Integer) = extractval(w.data[i])

@inline Base.getindex(A::Wishart3x3, ::LinearStorage, i::Integer) = @inbounds A.data[i]

@inline function Base.getindex(w::Wishart3x3, i::CartesianIndex{2})
    w[i[1], i[2]]
end
@inline function Base.getindex(iw::InverseWishart, i::Integer, j::Integer)
    i, j = minmax(i, j)
    @boundscheck (j > 3 || i < 1) && throwboundserror()
    im1 = i - 1
    @inbounds iw.data[3im1 - ( (im1*i) >> 1) + j].value
end
@inline function Base.getindex(w::Union{CholInvWishart{T},RevCholWishart{T}}, j::Integer, i::Integer) where T
    j < i && return zero(T)
    @boundscheck (j > 3 || i < 1) && throwboundserror()
    im1 = i - 1
    @inbounds extractval(w.data[3im1 - ( (im1*i) >> 1) + j])
end
function Base.show(io::IO, ::MIME"text/plain", x::InverseWishart{T}) where T
    denom = x[7] > 2 ? sqrt(x[7]-2) : one(T)
    println(io, "InverseWishart{$T} with ν = $(x[7]+2)")
    Base.print_matrix(io, x / denom)
end
function Base.show(io::IO, ::MIME"text/plain", x::CholInvWishart{T}) where T
    denom = x[7] > 2 ? sqrt(x[7]-2) : one(T)
    println(io, "Cholesky factor of InverseWishart{$T} with ν = $(x[7]+2)")
    Base.print_matrix(io, x / denom)
end

@inline function InverseWishart(x::T, y::T, z::T) where T
    InverseWishart(x * x, x * y, x * z, y * y, y * z, z * z, one(T), one(T))
end

@inline function Base.:+(a::W, b::W) where W <: Wishart3x3
    W(SIMDPirates.vadd(a.data, b.data))
end
@inline function Base.:*(L::Union{CholInvWishart{T},RevCholWishart{T}}, x::SVector{3,T}) where T
    SVector(
        L[1]*x[1],
        L[2]*x[1] + L[4]*x[2],
        L[3]*x[1] + L[5]*x[2] + L[6]*x[3]
    )
end
@inline function Base.:/(a::W, x::T) where {T <: Number, W <: Wishart3x3{T}}
    W(
        a[1] / x, a[2] / x, a[3] / x, a[4] / x, a[5] / x, a[6] / x, a[7], a[8]
    )
end

@inline extract_ν(iw::Union{InverseWishart,CholInvWishart}) = iw[7]
@generated function extract_ν(iw::AbstractVector{<:Union{InverseWishart,CholInvWishart}}, ::Val{NG}) where NG
    quote
        $(Expr(:meta, :inline))
        @ntuple $NG ng -> iw[ng][7]
    end
end
@inline extract_α(iw::Union{InverseWishart,CholInvWishart}) = iw[8]
@generated function extract_α(iw::AbstractVector{<:Union{InverseWishart,CholInvWishart}}, ::Val{NG}) where NG
    quote
        $(Expr(:meta, :inline))
        @ntuple $NG ng -> iw[ng][8]
    end
end

@inline function inv_and_cholesky!(riwv::AbstractVector{RevCholWishart{T}},
                                    ciwv::AbstractVector{CholInvWishart{T}},
                                    iwv::AbstractVector{InverseWishart{T}}) where T

    @inbounds for i ∈ eachindex(iwv)
        iw = iwv[i]
        @fastmath begin
            L11 = sqrt(iw[1])
            R11 = 1 / L11
            L21 = R11 * iw[2]
            L31 = R11 * iw[3]
            L22 = sqrt(iw[4] - L21^2)
            R22 = 1 / L22
            L32 = R22 * (iw[5] - L21 * L31)
            L33 = sqrt(iw[6] - L31^2 - L32^2)
            R33 = 1 / L33

            R21 = - R22 * L21 * R11
            R31 = - R33 * ( L31*R11 + L32*R21 )
            R32 = - R33 * L32 * R22
        end
        ciwv[i] = CholInvWishart(
            L11, L21, L31, L22, L32, L33, iw[7], iw[8]
        )
        riwv[i] = RevCholWishart(
            R11, R21, R31, R22, R32, R33
        )
    end
end
