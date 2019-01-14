
abstract type AbstractUpperTriangle{T} <: AbstractMatrix{T} end
abstract type AbstractSymmetric{T} <: AbstractMatrix{T} end

struct UpperTriangle2{T} <: AbstractUpperTriangle{T}
    data::SVector{3,T}
end
struct UpperTriangle3{T} <: AbstractUpperTriangle{T}
    data::SVector{6,T}
end
UpperTriangle2(A...) = UpperTriangle2(SVector{3}(A...))
UpperTriangle3(A...) = UpperTriangle3(SVector{6}(A...))

struct SymmetricM2{T} <: AbstractSymmetric{T}
    data::SVector{3,T}
end
struct SymmetricM3{T} <: AbstractSymmetric{T}
    data::SVector{6,T}
end
SymmetricM2(A...) = SymmetricM2(SVector{3}(A...))
SymmetricM3(A...) = SymmetricM3(SVector{6}(A...))
@inline Base.getindex(A::Union{<: AbstractUpperTriangle, AbstractSymmetric}, i) = @inbounds A.data[i]
@inline function Base.getindex(A::AbstractUpperTriangle, i, j)
    @boundscheck begin
        i1, j1 = minmax(i,j)
        (i1 < 1 || j1 > 3) && throwboundserror()
    end
    i > j && return 0.0
    @inbounds A.data[i + (j-1)*j ÷ 2]
end
@inline function Base.getindex(A::AbstractSymmetric, i, j)
    i, j = minmax(i, j)
    @inbounds A.data[i + (j-1)*j ÷ 2]
end
Base.size(::Union{UpperTriangle2,SymmetricM2}) = (2,2)
Base.size(::Union{UpperTriangle3,SymmetricM3}) = (3,3)
Base.length(::Union{UpperTriangle2,SymmetricM2}) = 4
Base.length(::Union{UpperTriangle3,SymmetricM3}) = 9
Base.Array(S::SymmetricM2{T}) where T = T[S[1] S[2]; S[2] S[3]]
Base.Array(U::UpperTriangle2{T}) where T = T[U[1] U[2] ; 0 U[3]]
Base.Array(S::SymmetricM3{T}) where T = T[S[1] S[2] S[4]; S[2] S[3] S[5]; S[4] S[5] S[6]]
Base.Array(U::UpperTriangle3{T}) where T = T[U[1] U[2] U[4]; 0 U[3] U[5]; 0 0 U[6]]

@inline function Base.:+(A1::T, A2::T) where {T <: Union{AbstractUpperTriangle,AbstractSymmetric}}
    T( A1.data + A2.data )
end

@inline function Base.:*(A::UpperTriangle2, B::UpperTriangle2)
    @fastmath UpperTriangle2(
        A[1]*B[1],
        A[1]*B[2] + A[2]*B[3],
        A[3]*B[3]
    )
end
@inline function Base.:*(A::UpperTriangle3, B::UpperTriangle3)
    @fastmath UpperTriangle3(
        A[1]*B[1],
        A[1]*B[2] + A[2]*B[3],
        A[3]*B[3],
        A[1]*B[4] + A[2]*B[5] + A[4]*B[6],
        A[3]*B[5] + A[5]*B[6],
        A[6]*B[6]
    )
end
@inline function Base.:*(a::UpperTriangle2, b::SVector{2})
    SVector(
        a[1]*b[1] + a[2]*b[2],
        a[3]*b[2]
    )
end
@inline function Base.:*(a::UpperTriangle3, b::SVector{3})
    SVector(
        a[1]*b[1] + a[2]*b[2] + a[4]*b[3],
        a[3]*b[2] + a[5]*b[3],
        a[6]*b[3]
    )
end
@inline function Base.:*(a::Adjoint{T,SVector{3,T}}, S::SymmetricM3{T}) where T
    @inbounds SVector(
        S[1]*a[1] + S[2]*a[2] + S[4]*a[3],
        S[2]*a[1] + S[3]*a[2] + S[5]*a[3],
        S[4]*a[1] + S[5]*a[2] + S[6]*a[3]
    )'
end
@inline function Base.:*(S::SymmetricM3, a::SVector{3})
    @inbounds SVector(
        S[1]*a[1] + S[2]*a[2] + S[4]*a[3],
        S[2]*a[1] + S[3]*a[2] + S[5]*a[3],
        S[4]*a[1] + S[5]*a[2] + S[6]*a[3]
    )
end
@inline function quadform(z::SVector{3}, Σ::SymmetricM3)
    @inbounds a = z[1]^2*Σ[1] + 2z[1]*z[2]*Σ[2] + 2z[1]*z[3]*Σ[4]
    @inbounds b = z[2]^2*Σ[3] + 2z[2]*z[3]*Σ[5] + z[3]^2*Σ[6]
    a + b
end
@inline function Base.:+(a::U, b::Number) where {U <: Union{AbstractSymmetric,AbstractUpperTriangle}}
    U(a.data + b)
end
@inline function Base.:+(a::Number, b::U) where {U <: Union{AbstractSymmetric,AbstractUpperTriangle}}
    U(a + b.data)
end
@inline function Base.:-(a::U, b::Number) where {U <: Union{AbstractSymmetric,AbstractUpperTriangle}}
    U(a.data - b)
end
@inline function Base.:-(a::Number, b::U) where {U <: Union{AbstractSymmetric,AbstractUpperTriangle}}
    U(a - b.data)
end
@inline function Base.:*(a::U, b::Number) where {U <: Union{AbstractSymmetric,AbstractUpperTriangle}}
    U(a.data * b)
end
@inline function Base.:*(a::Number, b::U) where {U <: Union{AbstractSymmetric,AbstractUpperTriangle}}
    U(a * b.data)
end
@inline function Base.:/(a::U, b::Number) where {U <: Union{AbstractSymmetric,AbstractUpperTriangle}}
    U(a.data / b)
end
@inline function Base.:/(a::Number, b::U) where {U <: Union{AbstractSymmetric,AbstractUpperTriangle}}
    U(a / b.data)
end
@inline function xxt(A::UpperTriangle2)
    SymmetricM2(
        A[1]^2 + A[2]^2,
        A[2]*A[3],
        A[3]^2
    )
end
@inline function xxt(A::UpperTriangle3)
    SymmetricM3(
        A[1]^2 + A[2]^2 + A[4]^2,
        A[2]*A[3] + A[4]*A[5],
        A[3]^2 + A[5]^2,
        A[4]*A[6],
        A[5]*A[6],
        A[6]^2
    )
end

function Base.inv(U::UpperTriangle2)
    @fastmath @inbounds begin
        t11 = 1 / U[1]
        t22 = 1 / U[3]
        t12 = - U[2] * t11 * t22
    end
    UpperTriangle2( t11, t12, t22 )
end
function Base.inv(U::UpperTriangle3)
    @fastmath @inbounds begin
        t11 = 1 / U[1]
        t22 = 1 / U[3]
        t33 = 1 / U[6]
        t12 = - U[2] * t11 * t22
        t13 = - (U[4] * t11 + U[5] * t12) * t33
        t23 = - U[5] * t22 *t33
    end
    UpperTriangle3( t11, t12, t22, t13, t23, t33 )
end

function revchol(S::SymmetricM2)
    @fastmath @inbounds begin
        ru22 = sqrt(S[3])
        ru12 = S[2] / ru22
        ru11 = sqrt(S[1] - ru12^2)
    end
    UpperTriangle2( ru11, ru12, ru22 )
end
function revchol(S::SymmetricM3)
    @fastmath @inbounds begin
        ru33 = sqrt(S[6])
        ru13 = S[4] / ru33
        ru23 = S[5] / ru33
        ru22 = sqrt(S[3] - ru23^2)
        ru12 = (S[2] - ru13*ru23) / ru22
        ru11 = sqrt(S[1] - ru12^2 - ru13^2)
    end
    UpperTriangle3( ru11, ru12, ru22, ru13, ru23, ru33 )
end

@generated function Base.:*(A::SMatrix{M,N,T}, B::SMatrix{N,P,T}) where {M,N,P,T}
    outtup = Vector{Expr}(undef, M*P)
    i = 0
    for p ∈ 1:P, m ∈ 1:M
        i += 1
        outtup[i] = :(@inbounds $(Symbol(:C_, p))[$m].value )
    end
    quote
        $(Expr(:meta, :inline))
        A_col = $(Expr(:tuple, [:(@inbounds Core.VecElement(A[$m,1])) for m ∈ 1:M]...))
        Base.Cartesian.@nexprs $P p -> C_p = SIMDPirates.extract_data(SIMDPirates.vmul(A_col, @inbounds B[1,p]))
        @inbounds for n ∈ 2:$N
            A_col = $(Expr(:tuple, [:(@inbounds Core.VecElement(A[$m,n])) for m ∈ 1:M]...))
            Base.Cartesian.@nexprs $P p -> C_p = SIMDPirates.vfma(A_col, (@inbounds B[n,p]), C_p)
        end
        SMatrix{$M,$P,$T}(
            $(Expr(:tuple, outtup...))
        )
    end

end
