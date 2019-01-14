
abstract type Wishart3x3{T} <: AbstractMatrix{T} end

struct InverseWishart{T} <: Wishart3x3{T}
    data::NTuple{8,VE{T}}
end
struct CholInvWishart{T} <: Wishart3x3{T}
    data::NTuple{8,T}
end
struct RevCholWishart{T} <: Wishart3x3{T}
    data::NTuple{8,T}
end

const WishartFactor{T} = Union{CholInvWishart{T},RevCholWishart{T}}

@inline InverseWishart(x::Vararg{T,8}   ) where T = InverseWishart{T}(VE.(x))
@inline InverseWishart{T}(x::Vararg{T,8}) where T = InverseWishart{T}(VE.(x))
# @generated function CholInvWishart(x::Vararg{T,8}) where T
#     quote
#         $(Expr(:meta,:inline))
#         CholInvWishart{$T}($(Expr(:tuple,[:(@inbounds x[$i]) for i ∈ 1:8]...)))
#     end
# end

@inline function CholInvWishart{T}(x_1::T,x_2::T,x_3::T,x_4::T,x_5::T,x_6::T) where T
    x_7 = x_8 = one(T)
    CholInvWishart{T}((Base.Cartesian.@ntuple 8 x))
end
@inline function CholInvWishart{T}(x_1::T,x_2::T,x_3::T,x_4::T,x_5::T,x_6::T,x_7::T,x_8::T) where T
    CholInvWishart{T}((Base.Cartesian.@ntuple 8 x))
end
@inline function CholInvWishart(x_1::T,x_2::T,x_3::T,x_4::T,x_5::T,x_6::T) where T
    x_7 = x_8 = one(T)
    CholInvWishart{T}((Base.Cartesian.@ntuple 8 x))
end
@inline function CholInvWishart(x_1::T,x_2::T,x_3::T,x_4::T,x_5::T,x_6::T,x_7::T,x_8::T) where T
    CholInvWishart{T}((Base.Cartesian.@ntuple 8 x))
end
@inline function RevCholWishart{T}(x_1::T,x_2::T,x_3::T,x_4::T,x_5::T,x_6::T) where T
    x_7 = x_8 = one(T)
    RevCholWishart{T}((Base.Cartesian.@ntuple 8 x))
end
@inline function RevCholWishart{T}(x_1::T,x_2::T,x_3::T,x_4::T,x_5::T,x_6::T,x_7::T,x_8::T) where T
    RevCholWishart{T}((Base.Cartesian.@ntuple 8 x))
end
@inline function RevCholWishart(x_1::T,x_2::T,x_3::T,x_4::T,x_5::T,x_6::T) where T
    x_7 = x_8 = one(T)
    RevCholWishart{T}((Base.Cartesian.@ntuple 8 x))
end
@inline function RevCholWishart(x_1::T,x_2::T,x_3::T,x_4::T,x_5::T,x_6::T,x_7::T,x_8::T) where T
    RevCholWishart{T}((Base.Cartesian.@ntuple 8 x))
end
# @inline RevCholWishart{T}(x::Vararg{T,8}) where T = RevCholWishart{T}(x)
# @inline RevCholWishart(x::Vararg{T,6}   ) where T = RevCholWishart{T}(x,T(1),T(1))
# @inline RevCholWishart{T}(x::Vararg{T,6}) where T = RevCholWishart{T}(x,T(1),T(1))
@inline CholInvWishart{T}(x::Vararg{SVec{W,T},8}) where {W,T} = CholInvWishart{SVec{W,T}}(x)

inv_type(::Type{CholInvWishart{T}}) where T = RevCholWishart{T}
inv_type(::Type{RevCholWishart{T}}) where T = CholInvWishart{T}
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
@inline Base.getindex(w::Wishart3x3, i::Integer) = extractval(@inbounds w.data[i])

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
@inline function Base.getindex(w::WishartFactor{T}, j::Integer, i::Integer) where T
    j < i && return zero(T)
    @boundscheck (j > 3 || i < 1) && throwboundserror()
    im1 = i - 1
    @inbounds extractval(w.data[3im1 - ( (im1*i) >> 1) + j])
end
function Base.show(io::IO, ::MIME"text/plain", x::InverseWishart{T}) where T
    denom = x[7] > 2 ? x[7]-2 : one(T)
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
@inline function Base.:*(L::WishartFactor{T}, x::SVector{3,T}) where T
    SVector(
        L[1]*x[1],
        L[2]*x[1] + L[4]*x[2],
        L[3]*x[1] + L[5]*x[2] + L[6]*x[3]
    )
end
# L2 is treated as a lower triangular matrix
@inline function Base.:*(L1::W, L2::NTuple{6,T}) where {T, W <: WishartFactor{T}}
    @fastmath W(
        L1[1]*L2[1],
        L1[2]*L2[1] + L1[4]*L2[2],
        L1[3]*L2[1] + L1[5]*L2[2] + L1[6]*L2[3],
        L1[4]*L2[4],
        L1[5]*L2[4] + L1[6]*L2[5],
        L1[6]*L2[6],
        L1[7],L1[8]
    )
end
@inline function Base.:/(a::W, x::T) where {T <: Number, W <: Wishart3x3{T}}
    xinv = 1 / x
    W(
        a[1] * xinv, a[2] * xinv, a[3] * xinv, a[4] * xinv, a[5] * xinv, a[6] * xinv, a[7], a[8]
    )
end

@inline extract_ν(iw::Union{InverseWishart,CholInvWishart}) = iw[7]
@generated function extract_ν(iw::AbstractVector{<:Union{InverseWishart{T},CholInvWishart{T}}}, ::Val{NG}) where {NG,T}
    quote
        $(Expr(:meta, :inline))
        ptr_iw = Base.unsafe_convert(Ptr{$T}, pointer(iw))
        $(Expr(:tuple, [:(unsafe_load(ptr_iw, $((ng-1)*8) + 7 ) ) for ng ∈ 1:NG]...))
    end
end
@inline extract_α(iw::Union{InverseWishart,CholInvWishart}) = iw[8]
@generated function extract_α(iw::AbstractVector{<:Union{InverseWishart{T},CholInvWishart{T}}}, ::Val{NG}) where {NG,T}
    quote
        $(Expr(:meta, :inline))
        ptr_iw = Base.unsafe_convert(Ptr{$T}, pointer(iw))
        $(Expr(:tuple, [:(unsafe_load(ptr_iw, $((ng-1)*8) + 8 ) ) for ng ∈ 1:NG]...))
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
            R11, R21, R31, R22, R32, R33, iw[7], iw[8]
        )
    end
end
@inline function triangle_inv(L::NTuple{8,T}) where T
    @inbounds @fastmath begin
        L11           = L[1]
        L21, L22      = L[2], L[4]
        L31, L32, L33 = L[3], L[5], L[6]
        R11 = 1 / L11
        R22 = 1 / L22
        R33 = 1 / L33
        R21 = - R22 * L21 * R11
        R31 = - R33 * ( L31*R11 + L32*R21 )
        R32 = - R33 * L32 * R22
    end
    @inbounds (R11, R21, R31, R22, R32, R33, L[7], L[8])
end
@inline function triangle_inv(L::NTuple{6,T}) where T
    @inbounds @fastmath begin
        L11           = L[1]
        L21, L22      = L[2], L[4]
        L31, L32, L33 = L[3], L[5], L[6]
        R11 = 1 / L11
        R22 = 1 / L22
        R33 = 1 / L33
        R21 = - R22 * L21 * R11
        R31 = - R33 * ( L31*R11 + L32*R21 )
        R32 = - R33 * L32 * R22
    end
    @inbounds (R11, R21, R31, R22, R32, R33)
end
@inline function Base.inv(w::W) where {T,W<:WishartFactor{T}}
    inv_type(W)(triangle_inv(w.data))
end

function Random.rand(rng::AbstractRNG, ciw::CholInvWishart{T}) where T

end
