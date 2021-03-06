
abstract type AbstractUpperTriangle{T} <: AbstractMatrix{T} end
# abstract type AbstractLowerTriangle{T} <: AbstractMatrix{T} end
abstract type AbstractSymmetric{T} <: AbstractMatrix{T} end

struct UpperTriangle2{T} <: AbstractUpperTriangle{T}
    data::SVector{3,T}
end
struct UpperTriangle3{T} <: AbstractUpperTriangle{T}
    data::SVector{6,T}
end
UpperTriangle2(A...) = UpperTriangle2(SVector{3}(A...))
UpperTriangle3(A...) = UpperTriangle3(SVector{6}(A...))
# struct LowerTriangle3{T} <: AbstractLowerTriangle{T}
#     data::SVector{6,T}
# end
# LowerTriangle3(A...) = LowerTriangle3(SVector{6}(A...))

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
function LinearAlgebra.logdet(U::UpperTriangle3)
    log(U[1]) + log(U[3]) + log(U[6])
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
    @fastmath @inbounds UpperTriangle2(
        A[1]*B[1],
        A[1]*B[2] + A[2]*B[3],
        A[3]*B[3]
    )
end
@inline function Base.:*(A::UpperTriangle3, B::UpperTriangle3)
    @fastmath @inbounds UpperTriangle3(
        A[1]*B[1],
        A[1]*B[2] + A[2]*B[3],
        A[3]*B[3],
        A[1]*B[4] + A[2]*B[5] + A[4]*B[6],
        A[3]*B[5] + A[5]*B[6],
        A[6]*B[6]
    )
end
# @inline function Base.:*(A::LowerTriangle3, B::LowerTriangle3)
#     @fastmath @inbounds LowerTriangle3(
#         A[1]*B[1],
#         A[1]*B[2] + A[2]*B[3],
#         A[3]*B[3],
#         A[1]*B[4] + A[2]*B[5] + A[4]*B[6],
#         A[3]*B[5] + A[5]*B[6],
#         A[6]*B[6]
#     )
# end
@inline function Base.:*(a::UpperTriangle2, b::SVector{2})
    @fastmath @inbounds SVector(
        a[1]*b[1] + a[2]*b[2],
        a[3]*b[2]
    )
end
@inline function Base.:*(a::UpperTriangle3, b::SVector{3})
    @fastmath @inbounds SVector(
        a[1]*b[1] + a[2]*b[2] + a[4]*b[3],
        a[3]*b[2] + a[5]*b[3],
        a[6]*b[3]
    )
end
@inline function Base.:*(a::Adjoint{T,SVector{3,T}}, S::SymmetricM3{T}) where T
    @fastmath @inbounds SVector(
        S[1]*a[1] + S[2]*a[2] + S[4]*a[3],
        S[2]*a[1] + S[3]*a[2] + S[5]*a[3],
        S[4]*a[1] + S[5]*a[2] + S[6]*a[3]
    )'
end
@inline function Base.:*(S::SymmetricM3, a::SVector{3})
    @fastmath @inbounds SVector(
        S[1]*a[1] + S[2]*a[2] + S[4]*a[3],
        S[2]*a[1] + S[3]*a[2] + S[5]*a[3],
        S[4]*a[1] + S[5]*a[2] + S[6]*a[3]
    )
end
@inline function quadform(z::SVector{3}, Σ::SymmetricM3)
    @inbounds @fastmath begin
         a = z[1]^2*Σ[1] + 2z[1]*z[2]*Σ[2] + 2z[1]*z[3]*Σ[4]
         b = z[2]^2*Σ[3] + 2z[2]*z[3]*Σ[5] + z[3]^2*Σ[6]
         a + b
     end
end
@inline function quadform(U::UpperTriangle3, Σ::SymmetricM3) # U * Σ * U'
    @fastmath begin
        @inbounds U11, U12, U22, U13, U23, U33 = U[1], U[2], U[3], U[4], U[5], U[6]
        @inbounds Σ11, Σ12, Σ22, Σ13, Σ23, Σ33 = Σ[1], Σ[2], Σ[3], Σ[4], Σ[5], Σ[6]

        UΣ12 = (U11*Σ12 + U12*Σ22 + U13*Σ23)
        UΣ13 = (U11*Σ13 + U12*Σ23 + U13*Σ33)
        S11 = (U11*Σ11 + U12*Σ12 + U13*Σ13) * U11 + UΣ12 * U12 + UΣ13 * U13
        S12 = UΣ12 * U22 + UΣ13 * U23
        S13 = UΣ13 * U33
        UΣ23 = U22*Σ23 + U23*Σ33
        S22 = (U22*Σ22 + U23*Σ23) * U22 + UΣ23 * U23
        S23 = UΣ23*U33
        S33 = U33*Σ33*U33
        SymmetricM3(
            S11, S12, S22, S13, S23, S33
        )
    end
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
    @inbounds SymmetricM2(
        A[1]^2 + A[2]^2,
        A[2]*A[3],
        A[3]^2
    )
end
@inline function xxt(A::UpperTriangle3)
    @inbounds SymmetricM3(
        A[1]^2 + A[2]^2 + A[4]^2,
        A[2]*A[3] + A[4]*A[5],
        A[3]^2 + A[5]^2,
        A[4]*A[6],
        A[5]*A[6],
        A[6]^2
    )
end
@inline function xtx(A::UpperTriangle2)
    @inbounds SymmetricM2(
        A[1]^2,
        A[1]*A[2],
        A[2]^2 + A[3]^2
    )
end
@inline function xtx(A::UpperTriangle3)
    @inbounds SymmetricM3(
        A[1]^2,
        A[1]*A[2],
        A[2]^2 + A[3]^2,
        A[1]*A[4],
        A[2]*A[4] + A[3]*A[5],
        A[4]^2 + A[5]^2 + A[6]^2
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

function revchol(S::SymmetricM2{T}) where {T}
    @fastmath @inbounds begin
        ru22 = finite_sqrt(S[3], T(1e-4))
        ru12 = S[2] / ru22
        ru11 = finite_sqrt(S[1] - ru12^2, T(1e-4))
    end
    UpperTriangle2( ru11, ru12, ru22 )
end
function revchol(S::SymmetricM3{T}) where {T}
    @fastmath @inbounds begin
        ru33 = finite_sqrt(S[6], T(1e-4))
        ru13 = S[4] / ru33
        ru23 = S[5] / ru33
        ru22 = finite_sqrt(S[3] - ru23^2, T(1e-4))
        ru12 = (S[2] - ru13*ru23) / ru22
        ru11 = finite_sqrt(S[1] - ru12^2 - ru13^2, T(1e-4))
    end
    UpperTriangle3( ru11, ru12, ru22, ru13, ru23, ru33 )
end
function chol(S::SymmetricM3{T}) where {T}
    @fastmath @inbounds begin
        U11 = finite_sqrt(S[1], T(1e-4))
        U12 = S[2] / U11
        U13 = S[4] / U11
        U22 = finite_sqrt(S[3] - U12*U12, T(1e-4))
        U23 = (S[5] - U12*U13) / U22
        U33 = finite_sqrt(S[6] - U13*U13 - U23*U23, T(1e-4))
        UpperTriangle3(
            U11, U12, U22, U13, U23, U33
        )
    end
end
function chol(S::SymmetricM2{T}) where {T}
    @fastmath @inbounds begin
        U11 = finite_sqrt(S[1], T(1e-4))
        U12 = S[2] / U11
        U22 = finite_sqrt(S[3] - U12*U12, T(1e-4))
        UpperTriangle2(
            U11, U12, U22
        )
    end
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
