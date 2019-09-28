


abstract type AbstractBandMatrix{T,B} <: AbstractMatrix{T} end

struct BandMatrix{T,B} <: AbstractBandMatrix{T,B}
    ncol::Base.RefValue{Int}
    data::Vector{T}
end
struct DualBandMatrix{T,B,P} <: AbstractBandMatrix{ForwardDiff.Dual{Nothing,T,P},B}
    bm::BandMatrix{T,B}
    partials::Vector{ForwardDiff.ForwardDiff.Partials{P,T}}
end

function BandMatrix{T,B}(::UndefInitializer, ncols) where {T,B}
    ncol = Ref{Int}(ncols)
    data = Vector{T}(undef, ncols*(B+1))
    # @show ncol, data
    BandMatrix{T,B}(ncol, data)
end
function DualBandMatrix{T,B,P}(::UndefInitializer, ncols) where {T,B,P}
    DualBandMatrix{T,B,P}(
        BandMatrix{T,B}(undef, ncols),
        Vector{ForwardDiff.ForwardDiff.Partials{P,T}}(undef, ncols*(B+1))
    )
end

Random.rand!(bm::BandMatrix) = (rand!(bm.data); bm)
Random.randn!(bm::BandMatrix) = (randn!(bm.data); bm)
Random.randexp!(bm::BandMatrix) = (randexp!(bm.data); bm)


ncol(A::BandMatrix) = A.ncol[]
ncol(A::DualBandMatrix) = A.bm.ncol[]
data(A::BandMatrix) = A.data
data(A::DualBandMatrix) = A.bm.data

Base.size(A::AbstractBandMatrix) = (nc = ncol(A); (nc, nc))
Base.length(A::AbstractBandMatrix) = (ncol(A)^2)

function Base.Array(A::AbstractBandMatrix{T,B}) where {T,B}
    N = ncol(A)
    out = zeros(T, N, N)
    for i ∈ 1:N-B-1
        for b ∈ i:i+B
            out[b,i] = A[b,i]
        end
    end
    for i ∈ N-B:N
        for j ∈ i:N
            out[j,i] = A[j,i]
        end
    end
    out
end

function Base.resize!(bm::BandMatrix{T,B}, N) where {T,B}
    L = (B+1)*N
    bm.ncol[] = N
    resize!(bm.data, L)
    bm
end
function Base.resize!(bm::DualBandMatrix{T,B}, N) where {T,B}
    L = (B+1)*N
    bm.bm.ncol[] = N
    resize!(bm.bm.data, L)
    resize!(bm.partials, L)
    bm
end


function band_sub2ind(::Val{B}, i, j) where {B}
    diff = i - j
    Bp1 = B + 1
    Bp1 - diff + Bp1 * (j - 1 + diff)
    # 1 + diff + Bp1 * (j - 1)
end
function Base.getindex(A::BandMatrix{T,B}, i, j) where {T,B}
    diff = i - j
    if diff < 0
        return zero(T)
    elseif diff > B
        return zero(T)
    end
    A.data[band_sub2ind(Val{B}(), i, j)]
end
function Base.getindex(A::DualBandMatrix{T,B,P}, i, j) where {T,B,P}
    diff = i - j
    if diff < 0
        return zero(T)
    elseif diff > B
        return zero(T)
    end
    ind = band_sub2ind(Val{B}(), i, j)
    ForwardDiff.Dual{Nothing,T,P}(A.bm.data[ind], A.partials[ind])
end
@noinline throwboundserror(a, i) = throw(BoundsError(a, i))
@inline function Base.getindex(A::BandMatrix{T,B}, ind) where {T,B}
    @boundscheck ind > length(A.data) && throwboundserror(A.data, ind)
    @inbounds A.data[ind]
end
@inline function Base.getindex(A::DualBandMatrix{T,B,P}, ind) where {T,B,P}
    @boundscheck ind > length(A.partials) && throwboundserror(A.bm.data, ind)
    @inbounds ForwardDiff.Dual{Nothing,T,P}(A.bm.data[ind], A.partials[ind])
end
@inline function Base.setindex!(A::BandMatrix{T,B}, v, ind) where {T,B}
    @boundscheck ind > length(A.data) && throwboundserror(A.data, ind)
    A.data[ind] = v
end
@inline function Base.setindex!(A::DualBandMatrix{T,B,P}, v::ForwardDiff.Dual{Nothing,T,P}, ind) where {T,B,P}
    @boundscheck ind > length(A.partials) && throwboundserror(A.bm.data, ind)
    @inbounds A.bm.data[ind] = v.value
    @inbounds A.partials[ind] = v.partials
    v
end
@inline function Base.setindex!(A::DualBandMatrix{T,B,P}, v::T, ind) where {T,B,P}
    @boundscheck ind > length(A.partials) && throwboundserror(A.bm.data, ind)
    @inbounds A.bm.data[ind] = v
    # @inbounds A.partials[ind] = v.partials
    v
end
@inline function Base.setindex!(A::DualBandMatrix{T,B,P}, v::ForwardDiff.Partials{P,T}, ind) where {T,B,P}
    @boundscheck ind > length(A.partials) && throwboundserror(A.bm.data, ind)
    @inbounds A.partials[ind] = v.partials
    v
end

@inline function Base.setindex!(A::BandMatrix{T,B}, v, i, j) where {T,B}
    ind = band_sub2ind(Val{B}(), i, j)
    @boundscheck ind > length(A.data) && throwboundserror(A.data, ind)
    A.data[ind] = v
end
@inline function Base.setindex!(A::DualBandMatrix{T,B,P}, v::ForwardDiff.Dual{Nothing,T,P}, i, j) where {T,B,P}
    ind = band_sub2ind(Val{B}(), i, j)
    @boundscheck ind > length(A.partials) && throwboundserror(A.bm.data, ind)
    @inbounds A.bm.data[ind] = v.value
    @inbounds A.partials[ind] = v.partials
    v
end
@inline function Base.setindex!(A::DualBandMatrix{T,B,P}, v::T, i, j) where {T,B,P}
    ind = band_sub2ind(Val{B}(), i, j)
    @boundscheck ind > length(A.partials) && throwboundserror(A.bm.data, ind)
    @inbounds A.bm.data[ind] = v
    # @inbounds A.partials[ind] = v.partials
    v
end
@inline function Base.setindex!(A::DualBandMatrix{T,B,P}, v::ForwardDiff.Partials{P,T}, i, j) where {T,B,P}
    ind = band_sub2ind(Val{B}(), i, j)
    @boundscheck ind > length(A.partials) && throwboundserror(A.bm.data, ind)
    @inbounds A.partials[ind] = v.partials
    v
end

struct DualVector{T,P} <: AbstractVector{ForwardDiff.Dual{Nothing,T,P}}
    data::Vector{T}
    partials::Vector{ForwardDiff.ForwardDiff.Partials{P,T}}
end
function DualVector{T,P}(::UndefInitializer, L) where {T,P}
    DualVector{T,P}(
        Vector{T}(undef, L),
        Vector{ForwardDiff.Partials{P,T}}(undef, L)
    )
end
@inline Base.size(v::DualVector) = size(v.data)
@inline Base.length(v::DualVector) = length(v.data)
function Base.resize!(v::DualVector, N)
    resize!(v.data, N)
    resize!(v.partials, N)
    v
end

@inline function Base.getindex(A::DualVector{T,P}, ind) where {T,P}
    @boundscheck ind > length(A.partials) && throwboundserror(A.bm.data, ind)
    @inbounds ForwardDiff.Dual{Nothing,T,P}(A.data[ind], A.partials[ind])
end
@inline function Base.setindex!(A::DualVector{T,P}, v::ForwardDiff.Dual{Nothing,T,P}, ind) where {T,P}
    @boundscheck ind > length(A.partials) && throwboundserror(A.data, ind)
    @inbounds A.data[ind] = v.value
    @inbounds A.partials[ind] = v.partials
    v
end
@inline function Base.setindex!(A::DualVector{T,P}, v::T, ind) where {T,P}
    @boundscheck ind > length(A.partials) && throwboundserror(A.data, ind)
    @inbounds A.data[ind] = v
    # @inbounds A.partials[ind] = v.partials
    v
end
@inline function Base.setindex!(A::DualVector{T,P}, v::ForwardDiff.Partials{P,T}, ind) where {T,P}
    @boundscheck ind > length(A.partials) && throwboundserror(A.data, ind)
    @inbounds A.partials[ind] = v.partials
    v
end

Base.IndexStyle(::Type{<:DualVector}) = IndexLinear()

function self_dot(v::AbstractVector{T}) where {T}
    out = zero(eltype(v))
    @inbounds @simd for i ∈ eachindex(v)
        out += v[i] * v[i]
    end
    out
end
@generated function self_dot(v::DualVector{T,P}) where {T,P}
    quote
        real = self_dot(v.data)
        Base.Cartesian.@nexprs $P p -> d_p = zero(T)
        # L = length(v.partials)
        # @inbounds @fastmath for i ∈ 0:4:L-4
        #     Base.Cartesian.@nexprs 4 j -> begin
        #         scale_j = 2v.data[i+j]
        #         Base.Cartesian.@nexprs $P p -> d_p += v.partials[i+j][p] * scale_j
        #     end
        # end
        # @inbounds @fastmath for i ∈ L+1-(L&3):L
        #     scale = 2v.data[i]
        #     Base.Cartesian.@nexprs $P p -> d_p += v.partials[i][p] * scale
        # end
        @inbounds @fastmath for i ∈ eachindex(v.partials)
            scale = 2v.data[i]
            Base.Cartesian.@nexprs $P p -> d_p += v.partials[i][p] * scale
        end
        ForwardDiff.Dual(real, ForwardDiff.Partials((Base.Cartesian.@ntuple $P p -> d_p)))
    end
end


Rsym(i, j) = Symbol(:R_, i, :_, j)
Lsym(i, j) = Symbol(:L_, i, :_, j)
Ssym(i, j) = Symbol(:S_, i, :_, j)
sym(S, i, j) = Symbol(S, :_, i, :_, j)


function precision_band_iteration(B, Ldata = :L, ind_expr = :($(B+1)*n), preloop = false, T = Float64)
    Bp1 = B + 1
    q = quote
        $(sym(:R,Bp1,Bp1)) = one($T)
    end
    # Load last row
    for b ∈ 1:B
        push!(q.args, :( $(sym(:L,Bp1, b)) = $Ldata[$ind_expr + $b] ))
    end
    # Calculate that last row from the inverse using preceding rows of inverse
    push!(q.args, :( $(sym(:R, Bp1, B)) = - $(sym(:L,Bp1,B)) * $(sym(:R,B,B)) ))
    for b ∈ 2:B
        push!(q.args, :( $(sym(:R, Bp1, Bp1-b)) = - (
            $(Expr(:call, :+, [:( $(sym(:L,Bp1,Bp1-bᵢ))*$(sym(:R,Bp1-bᵢ,Bp1-b)) ) for bᵢ ∈ 1:b]...))
        ) ))
    end
    # first colum of S equals first column of R
    for b ∈ 2:Bp1
        push!(q.args, :( $(sym(:S,b,1)) = $(sym(:R,b,1)) ))
    end
    for bcol ∈ 2:B # col
        for brow ∈ bcol+1:B+1 # row
            push!(q.args, :( $(sym(:S, brow, bcol)) =
                $(Expr(:call, :+, [:($(sym(:R,brow,bᵢ))*$(sym(:R,bcol,bᵢ))) for bᵢ ∈ 1:bcol]...))
            ))
        end
    end
    # push!(q.args, :( @show $(Expr(:tuple, [ :(abs2($(sym(:R,Bp1,b)))) for b ∈ 1:Bp1 ]...)) ))
    push!(q.args, :( $(sym(:S,Bp1,Bp1)) = $(Expr(:call,:+,1,[ :(abs2($(sym(:R,Bp1,b)))) for b ∈ 1:B ]...)) ) )
    push!(q.args, :( $(sym(:rS,Bp1,Bp1)) = inv(sqrt($(sym(:S,Bp1,Bp1))))))
    for b ∈ 1:B
        push!(q.args, :( $(sym(:S,Bp1,b)) *= $(sym(:rS,Bp1,Bp1))))
    end
    if preloop
        B += 1
        adjust = -1
    else
        adjust = 0
        Rtemp = :_R
        Rtempinv = :_iR
        for b ∈ 2:Bp1
            push!(q.args, :( $(sym(Rtemp,b,1)) = $(sym(:S,b,1)) ))
        end
        push!(q.args, :($(sym(Rtemp,2,2)) = sqrt( one($T) - abs2($(sym(Rtemp,2,1)) ))))
        push!(q.args, :($(sym(Rtempinv,2,2)) = inv($(sym(Rtemp,2,2)))))
        for b ∈ 3:Bp1
            push!(q.args, :($(sym(Rtemp,b,2)) = ( $(sym(:S,b,2)) - $(sym(Rtemp,b,1))*$(sym(Rtemp,2,1)) ) * $(sym(Rtempinv,2,2)) ))
        end
        for b ∈ 3:B
            # push!(q.args, :(@show $(Expr(:tuple, [:($(sym(Rtemp,b,bᵢ))*$(sym(Rtemp,b,bᵢ))) for bᵢ ∈ 1:b-1]...))))
            push!(q.args, :(
                $(sym(Rtemp,b,b)) = sqrt( one($T) - $(Expr(:call, :+, [:($(sym(Rtemp,b,bᵢ))*$(sym(Rtemp,b,bᵢ))) for bᵢ ∈ 1:b-1]...)))
            ))
            push!(q.args, :(
                $(sym(Rtempinv,b,b)) = inv($(sym(Rtemp,b,b)))
            ))
            for brow ∈ b+1:Bp1
                push!(q.args, :($(sym(Rtemp,brow,b)) = ( $(sym(:S,brow,b)) -
                    $(Expr(:call, :+, [:($(sym(Rtemp,brow,bᵢ))*$(sym(Rtemp,b,bᵢ))) for bᵢ ∈ 1:b-1]...)))
                    * $(sym(Rtempinv,b,b))
                ) )
            end
        end
        # for bᵢ ∈ 1:B
        #     push!(q.args, :(@show $(sym(Rtemp,Bp1,bᵢ))))
        # end
        push!(q.args, :(
            $(sym(Rtemp,Bp1,Bp1)) = sqrt( one($T) - $(Expr(:call, :+, [:(abs2($(sym(Rtemp,Bp1,bᵢ)))) for bᵢ ∈ 1:B]...)))
        ))
        push!(q.args, :(
            $(sym(:L,Bp1,Bp1)) = $Ldata[$ind_expr + $Bp1] = last_diag = $(sym(Rtempinv,Bp1,Bp1)) = inv($(sym(Rtemp,Bp1,Bp1)))
        ))
    end

    for b ∈ 2:B
        push!(q.args, :( $(sym(:R,b,1)) = $(sym(:S,b+1+adjust,2+adjust)) ))
    end

    # push!(q.args, :(@show $(sym(:R,2,1))))

    push!(q.args, :( $(sym(:R,2,2)) = sqrt( one($T) - abs2($(sym(:R,2,1))) ) ))
    push!(q.args, :( $(sym(:iR,2,2)) = inv($(sym(:R,2,2))) ))
    for b ∈ 3:B
        push!(q.args, :( $(sym(:R,b,2)) = ($(sym(:S,b+1+adjust,3+adjust)) - $(sym(:R,2,1))*$(sym(:R,b,1))) * $(sym(:iR,2,2)) ))
    end
    for bcol ∈ 3:B
        push!(q.args, :( $(sym(:R,bcol,bcol)) = sqrt(one($T) - $(Expr(:call,:+,[:(abs2($(sym(:R,bcol,bᵢ)))) for bᵢ ∈ 1:bcol-1]...))) ) )
        if bcol < B
            push!(q.args, :( $(sym(:iR,bcol,bcol)) = inv($(sym(:R,bcol,bcol))) ))
        end
        for brow ∈ bcol+1:B
            push!(q.args, :(
                $(sym(:R,brow,bcol)) = ( $(sym(:S,brow+1+adjust,bcol+1+adjust)) - $(Expr(:call,:+,[:($(sym(:R,bcol,bᵢ))*$(sym(:R,brow,bᵢ))) for bᵢ ∈ 1:bcol-1]...)) ) * $(sym(:iR,bcol,bcol))
            ) )
        end
    end

    if preloop
        push!(q.args, :(
            $(sym(:L,Bp1,Bp1)) = $Ldata[$ind_expr + $B] = last_diag = inv($(sym(:R,B,B)))
        ))
    end
    q
end

# Assumes R is
# function fill_precision_factor!(L::BandMatrix{T,B}) where {T,B}
@generated function fill_precision_factor!(L::AbstractBandMatrix{T,B}) where {T,B}
    Bp1 = B + 1
    q = quote
        L[$Bp1] = one($T)
        R_1_1 = 1
        L_2_1 = L[$(Bp1 + B)]
        iR_2_2 = sqrt(1 + L_2_1^2)
        L[$(2Bp1)] = iR_2_2
        R_2_2 = 1 / iR_2_2
        R_2_1 = - L_2_1 * R_2_2
        # @show R_2_1
        # @show L_2_1
        # println("hi")
    end
    for b ∈ 2:B-1
        push!(q.args, precision_band_iteration(b, :L, Bp1*b + Bp1 - b - 1, true, T))
        # push!(q.args,
        #     quote
        #         Smat = SMatrix{3,3}(1.0, S_2_1, S_3_1,
        #                             S_2_1, (R_2_1^2 + R_2_2^2), S_3_2,
        #                             S_3_1, S_3_2, 1.0)
        #         SmatChol = cholesky(Smat).L
        #         Rmat = SMatrix{3,3}(1.0, 0.0, 0.0,
        #                             R_2_1, R_2_2, 0.0,
        #                             R_3_1, R_3_2, R_3_3)'
        #         println("Smat: ")
        #         display(Smat)
        #         println("Smatchol: ")
        #         display(SmatChol)
        #         println("Rmat: ")
        #         display(Rmat)
        #     end)
    end
    push!(q.args,
        quote
            for n ∈ $(B):L.ncol[]-1
                $(precision_band_iteration(B, :L, :($Bp1*n), false, T))
                # Smat = SMatrix{4,4}(1.0, S_2_1, S_3_1, S_4_1,
                #                     S_2_1, (R_2_1^2 + R_2_2^2), S_3_2, S_4_2,
                #                     S_3_1, S_3_2, (R_3_1^2 + R_3_2^2 + R_3_3^2), S_4_3,
                #                     S_4_1, S_4_2, S_4_3, 1.0)
                # SmatChol = cholesky(Smat).L
                # Rmat = SMatrix{4,4}(1.0, 0.0, 0.0, 0.0,
                #                     _R_2_1, _R_2_2, 0.0, 0.0,
                #                     _R_3_1, _R_3_2, _R_3_3, 0.0,
                #                     _R_4_1, _R_4_2, _R_4_3, _R_4_4)'
                # println("Smat: ")
                # display(Smat)
                # println("Smatchol: ")
                # display(SmatChol)
                # println("Rmat: ")
                # display(Rmat)
            end
        end
    )

    quote
        @fastmath @inbounds begin
        # begin
            $q
        end
    end
end

# Assumes R is
# function fill_precision_factor!(L::BandMatrix{T,B}) where {T,B}
@generated function fill_precision_factor_det!(L::AbstractBandMatrix{T,B}) where {T,B}
    Bp1 = B + 1
    q = quote
        L[$Bp1] = one($T)
        R_1_1 = 1
        L_2_1 = L[$(Bp1 + B)]
        iR_2_2 = sqrt(1 + L_2_1^2)
        L[$(2Bp1)] = iR_2_2
        R_2_2 = 1 / iR_2_2
        R_2_1 = - L_2_1 * R_2_2
        log_det = log(iR_2_2)
    end
    for b ∈ 2:B-1
        push!(q.args, precision_band_iteration(b, :L, Bp1*b + Bp1 - b - 1, true, T))
        push!(q.args, :(log_det += log(last_diag)))
    end
    push!(q.args,
        quote
            for n ∈ $(B):L.ncol[]-1
                $(precision_band_iteration(B, :L, :($Bp1*n), false, T))
                log_det += log(last_diag)
            end
        end
    )

    quote
        @fastmath @inbounds begin
        # begin
            $q
        end
        log_det
    end
end



# Assumes R is
# function fill_precision_factor!(L::BandMatrix{T,B}) where {T,B}
@generated function fill_precision_factor!(y, L::AbstractBandMatrix{T,B}, x) where {T,B}
    Bp1 = B + 1
    q = quote
        L[$Bp1] = one($T)
        R_1_1 = 1
        L_2_1 = L[$(Bp1 + B)]
        iR_2_2 = sqrt(1 + L_2_1^2)
        L[$(2Bp1)] = iR_2_2
        R_2_2 = 1 / iR_2_2
        R_2_1 = - L_2_1 * R_2_2
        # det = iR_2_2
        y[1] = (eltype(y))(x[1])
        y[2] = x[2] * iR_2_2 + x[1] * L_2_1
    end
    for b ∈ 2:B-1
        push!(q.args, precision_band_iteration(b, :L, Bp1*b + Bp1 - b - 1, true, T))
        # push!(q.args, :(det += last_diag))
        push!(q.args, :(y[$(b+1)] = $(Expr(:call,:+,[:($(sym(:L,b+1,bᵢ))*x[$(bᵢ)]) for bᵢ ∈ 1:b+1]...))))
    end
    push!(q.args,
        quote
            for n ∈ $(B):L.ncol[]-1
                $(precision_band_iteration(B, :L, :($Bp1*n), false, T))
                # det += last_diag
                push!(q.args, :(y[$(b+1)] = $(Expr(:call,:+,[:($(sym(:L,Bp1,bᵢ))*x[n+$(bᵢ - B)]) for bᵢ ∈ 1:Bp1]...))))
            end
        end
    )

    quote
        @fastmath @inbounds begin
        # begin
            $q
        end
    end
end

# Assumes R is
# function fill_precision_factor!(L::BandMatrix{T,B}) where {T,B}
@generated function fill_precision_factor_det!(y, L::AbstractBandMatrix{T,B}, x) where {B,T}
    Bp1 = B + 1
    q = quote
        L[$Bp1] = one($T)
        R_1_1 = 1
        L_2_1 = L[$(Bp1 + B)]
        iR_2_2 = sqrt(1 + L_2_1^2)
        L[$(2Bp1)] = iR_2_2
        R_2_2 = 1 / iR_2_2
        R_2_1 = - L_2_1 * R_2_2
        log_det = log(iR_2_2)
        y[1] = (eltype(y))(x[1])
        y[2] = x[2] * iR_2_2 + x[1] * L_2_1
    end
    for b ∈ 2:B-1
        push!(q.args, precision_band_iteration(b, :L, Bp1*b + Bp1 - b - 1, true, T))
        push!(q.args, :(log_det += log(last_diag)))
        push!(q.args, :(y[$(b+1)] = $(Expr(:call,:+,[:($(sym(:L,b+1,bᵢ))*x[$(bᵢ)]) for bᵢ ∈ 1:b+1]...))))
    end
    push!(q.args,
        quote
            for n ∈ $(B):ncol(L)-1
                $(precision_band_iteration(B, :L, :($Bp1*n), false, T))
                log_det += log(last_diag)
                # if any(isnan, log_det.partials.values)
                #     @show last_diag
                #     @show log_det
                # end
                y[n+1] = $(Expr(:call,:+,[:($(sym(:L,Bp1,bᵢ))*x[n+$(bᵢ - B)]) for bᵢ ∈ 1:Bp1]...))
            end
        end
    )

    quote
        @fastmath @inbounds begin
        # begin
            $q
        end
        log_det
    end
end


mutable struct InvCholCovar{T,B,P,R,L,O} <: DifferentiableObjects.AbstractDifferentiableObject{P,T}
    bandmat::DualBandMatrix{T,B,P}
    y::DualVector{T,P}
    x::Vector{T}
    δt::Vector{T}
    state::DifferentiableObjects.BFGSState{P,T,R,L}
    initial_x::PaddedMatrices.MutableFixedSizeVector{P,T,R,R}
    ls::DifferentiableObjects.BackTracking2{O,T,Int}
    ∇::PaddedMatrices.MutableFixedSizeVector{P,T,R,R}
end

@generated function InvCholCovar{T,B}(N::Integer) where {T,B}
    P = 2B
    quote
        InvCholCovar(
            DualBandMatrix{$T,$B,$P}(undef, N),
            DualVector{$T,$P}(undef, N),
            Vector{$T}(undef, $B*N),
            Vector{$T}(undef, $B*N),
            DifferentiableObjects.BFGSState(Val($P), $T),
            fill!(PaddedMatrices.MutableFixedSizeVector{$P,$T}(undef), zero($T)),
            DifferentiableObjects.BackTracking2{$T}(Val(2)),
            PaddedMatrices.MutableFixedSizeVector{$P,$T}(undef)
        )
    end
end
@generated function InvCholCovar{T,B}(x) where {T,B}
    P = 2B
    quote
        N = length(x)
        InvCholCovar(
            DualBandMatrix{$T,$B,$P}(undef, N),
            DualVector{$T,$P}(undef, N),
            x,
            Vector{$T}(undef, $B*N),
            DifferentiableObjects.BFGSState(Val($P), $T),
            fill!(PaddedMatrices.MutableFixedSizeVector{P,T,R,R}{$P,$T}(undef), zero($T)),
            DifferentiableObjects.BackTracking2{$T}(Val(2)),
            PaddedMatrices.MutableFixedSizeVector{P,T,R,R}{$P,$T}(undef)
        )
    end
end
function InvCholCovar{T,B}(x, t) where {T,B}
    icc = InvCholCovar{T,B}(x)
    fill_δt!(icc.δt, t, Val(B))
    icc
end

function Base.resize!(icc::InvCholCovar{T,B,P}, N) where {T,B,P}
    resize!(icc.bandmat, N)
    resize!(icc.y, N)
    resize!(icc.x, N)
    resize!(icc.δt, N*B)
    icc
end


function fit!(icc::InvCholCovar{T}) where {T}
    DifferentiableObjects.optimize_scale!(icc.state, icc, icc.initial_x, icc.ls, T(10), T(1e-3))
end


@generated function fill_δt!(δt, t, ::Val{B}) where B
    Bp1 = B + 1
    quote
        Base.Cartesian.@nexprs $B i -> begin
            tᵢ = t[i+1]
            Base.Cartesian.@nexprs i b -> (δt[i*$B + $B - i + b] = t[b] - tᵢ)
        end
        for i ∈ 1:length(t) - $Bp1
            tᵢ = t[i+$Bp1]
            Base.Cartesian.@nexprs $B b -> (δt[i*$B + $(B*B) + b] = t[i+b] - tᵢ)
        end
    end
end

function decorrelate_data!(icc::InvCholCovar{T,B,P}, X::AbstractMatrix, t) where {T,B,P}
    N, M = size(X)
    resize!(icc, N)
    fill_δt!(icc.δt, t, Val(B))
    for m ∈ 1:M
        allfinite = false
        for i ∈ 1:100
        # icc.x .= @view X[:,m]
            stdx = std(@view(X[:,m]))
            invstdx = 1 / stdx
            @inbounds @simd ivdep for i in 1:size(X,1)
                icc.x[i] = X[i,m] * invstdx
            end
            DifferentiableObjects.optimize_scale!(icc.state, icc, icc.initial_x, icc.ls, T(10), 1e-3)
            allfinite = all(isfinite, icc.y.data)
            if !allfinite
                Random.randn!(icc.initial_x)
                continue
            end
            @inbounds @simd ivdep for i in 1:size(X,1)
                X[i,m] = icc.y.data[i] * stdx
            end
            icc.initial_x .= zero(T)
            break
        end
        allfinite || throw("Not all finite after 100 attempts.")
    end
end
function decorrelate_data!(icc::InvCholCovar{T,B,P}, X::AbstractVector, t) where {T,B,P}
    N = length(X)
    resize!(icc, N)
    fill_δt!(icc.δt, t, Val(B))
    stdx = std(X)
    invstdx = 1 / stdx
    if isa(X, Vector{T})
        icc.x = X
        icc.x .*= invstdx
    else
        icc.x .= X .* invstdx
    end
    DifferentiableObjects.optimize_scale!(icc.state, icc, icc.initial_x, icc.ls, T(10), 1e-3)
    X .= icc.y.data .* stdx
end


@generated zero_tuple(::Val{N}, ::Type{T}) where {N,T} = Expr(:tuple, [zero(T) for n ∈ 1:N]...)
@inline function inv_sigmoid(x)
    expm1x = expm1(x)
    expm1x/(2+expm1x)
end
@inline function ∂inv_sigmoid(x)
    expx = exp(x)
    isx = (expx-1)/(expx+1)
    ∂isx = 2expx/(expx+1)^2
    isx, ∂isx
end

@generated function DifferentiableObjects.fdf(icc::InvCholCovar{T,B,P}, θ::AbstractVector{T}) where {T,B,P}
    Bp1 = B+1
    quote
        # Σθ² = zero(T)
        Base.Cartesian.@nexprs $B b -> begin
            @inbounds θ_b = θ[b]
            @inbounds θ_{$B+b} = exp(θ[$B+b])
            # signθ_b = sign(θ_b)
            # absθ_b, ∂absθ_b = ∂inv_sigmoid(abs(θ_b))
            # Σθ² = fma(θ_b, θ_b, Σθ²)
            # icc.∇[b] = θ_b + θ_b
        end
        δta = icc.δt
        @inbounds  begin
            Base.Cartesian.@nexprs $B i -> begin
                # tᵢ = t[i]
                Base.Cartesian.@nexprs i b -> begin
                    δt = δta[i*$B + $B - i + b]
                    # v1 = absθ_{$B - i + b}^(δt-1)
                    # v = copysign(absθ_{$B - i + b} * v1, signθ_{$B - i + b})
                    # p = δt * v1 * ∂absθ_{$B - i + b}
                    # icc.bandmat[i*$Bp1 + $B - i + b] = ForwardDiff.Dual(v,ForwardDiff.Partials(Base.setindex(zero_tuple(Val(P), T), p, $B - i + b)))

                    ∂ϕ_b = exp(δt * θ_{$P - i + b})
                    ϕ_b = θ_{$B - i + b} * ∂ϕ_b
                    ∂ϕ_{$B+b} = ϕ_b * δt * θ_{$P - i + b}

                    ∂ϕ_btup = Base.setindex(Base.setindex(zero_tuple(Val(P), T), ∂ϕ_b, $B - i + b), ∂ϕ_{$B+b}, $P - i + b)
                    icc.bandmat[i*$Bp1 + $B - i + b] = ForwardDiff.Dual(ϕ_b, ForwardDiff.Partials(∂ϕ_btup))
                end
            end
        end
        @inbounds for i ∈ 1:length(icc.x) - $Bp1
            # tᵢ = t[i+$Bp1]
            Base.Cartesian.@nexprs $B b -> begin
                δt = δta[i*$B + $(B*B) + b]
                # v1 = absθ_b^(δt-1)
                # v = copysign(absθ_b * v1, signθ_b)
                # p = δt * v1 * ∂absθ_b
                # icc.bandmat[i*$Bp1 + $(Bp1*B) + b] = ForwardDiff.Dual(v,ForwardDiff.Partials(Base.setindex(zero_tuple(Val(P), T), p, b)))
                # ϕ_b, ∂ϕ_b = ∂inv_sigmoid( θ_b*δt )
                # icc.bandmat[i*$Bp1 + $(Bp1*B) + b] = ForwardDiff.Dual(ϕ_b,ForwardDiff.Partials(Base.setindex(zero_tuple(Val(P), T), ∂ϕ_b * δt, b)))
                ∂ϕ_b = exp(δt * θ_{$B + b})
                ϕ_b = θ_b * ∂ϕ_b
                ∂ϕ_{$B+b} = ϕ_b * δt * θ_{$B + b}
                ∂ϕ_btup = Base.setindex(Base.setindex(zero_tuple(Val(P), T), ∂ϕ_b, b), ∂ϕ_{$B+b}, $B + b)
                icc.bandmat[i*$Bp1 + $(Bp1*B) + b] = ForwardDiff.Dual(ϕ_b,ForwardDiff.Partials(∂ϕ_btup))
            end
        end
        logdet = fill_precision_factor_det!(icc.y, icc.bandmat, icc.x)
        out = self_dot(icc.y) - logdet
        Base.Cartesian.@nexprs $P p -> @inbounds icc.∇[p] = out.partials[p]
        out.value #+ Σθ²
    end
end
@generated function DifferentiableObjects.f(icc::InvCholCovar{T,B,P}, θ::AbstractVector{T}) where {T,B,P}
    Bp1 = B+1
    quote
        # Σθ² = zero(T)
        Base.Cartesian.@nexprs $B b -> begin
            @inbounds θ_b = θ[b]
            @inbounds θ_{$B+b} = exp(θ[$B+b])
            # signθ_b = sign(θ_b)
            # absθ_b = abs(inv_sigmoid(θ_b))
            # Σθ² = fma(θ_b, θ_b, Σθ²)
        end
        # @show length(icc.bandmat.bm.data)
        # t = icc.t
        δta = icc.δt
        @inbounds begin
            Base.Cartesian.@nexprs $B i -> begin
                # tᵢ = t[i]
                Base.Cartesian.@nexprs i b -> begin
                    δt = δta[i*$B + $B - i + b]
                    # v = copysign(absθ_{$B - i + b}^δt, signθ_{$B - i + b})
                    # icc.bandmat.bm[i*$Bp1 + $B - i + b] = v
                    v = θ_{$B - i + b} * exp(δt * θ_{$P - i + b})
                    icc.bandmat.bm[i*$Bp1 + $B - i + b] = v# inv_sigmoid(θ_{$B - i + b} * δt)
                end
            end
        end
        @inbounds for i ∈ 1:length(icc.x) - $Bp1
            # tᵢ = t[i+$Bp1]
            Base.Cartesian.@nexprs $B b -> begin
                δt = δta[i*$B + $(B*B) + b]
                # v = copysign(absθ_b^δt, signθ_b)
                # icc.bandmat.bm[i*$Bp1 + $(Bp1*B) + b] = v
                v = θ_b * exp(δt * θ_{$B + b})
                icc.bandmat.bm[i*$Bp1 + $(Bp1*B) + b] = v #inv_sigmoid(θ_b * δt)
            end
        end
        logdet = fill_precision_factor_det!(icc.y.data, icc.bandmat.bm, icc.x)
        self_dot(icc.y.data) - logdet
    end
end

function DifferentiableObjects.scale_fdf(icc::InvCholCovar{T}, θ::AbstractVector{T}, scale_target) where {T}
    fval = DifferentiableObjects.fdf(icc, θ)
    scale = min(one(T), scale_target / norm(icc.∇))
    @inbounds @simd for i ∈ eachindex(icc.∇)
        icc.∇[i] *= scale
    end
    fval * scale, scale
end
function DifferentiableObjects.scaled_fdf(icc::InvCholCovar{T}, θ::AbstractVector{T}, scale) where {T}
    fval = DifferentiableObjects.fdf(icc, θ)
    @inbounds @simd for i ∈ eachindex(icc.∇)
        icc.∇[i] *= scale
    end
    fval * scale
end
DifferentiableObjects.gradient(icc::InvCholCovar) = icc.∇



@generated function logdettest(icc::InvCholCovar{T,B,P}, θ::AbstractVector{T}) where {T,B,P}
    Bp1 = B+1
    quote
        Base.Cartesian.@nexprs $B b -> begin
            @inbounds θ_b = θ[b]
            signθ_b = sign(θ_b)
            absθ_b, ∂absθ_b = ∂inv_sigmoid(abs(θ_b))
        end
        δta = icc.δt
        @inbounds  begin
            Base.Cartesian.@nexprs $B i -> begin
                # tᵢ = t[i]
                Base.Cartesian.@nexprs i b -> begin
                    # δt = tᵢ - t[b]
                    δt = δta[i*$B + $B - i + b]
                    v1 = absθ_{$B - i + b}^(δt-1)
                    v = copysign(absθ_{$B - i + b} * v1, signθ_{$B - i + b})
                    p = δt * v1 * ∂absθ_{$B - i + b}
                    # v = - exp(θ_b * δt)
                    # p = v * δt
                    # dual =
                    # @show dual
                    # ForwardDiff.Dual(v,ForwardDiff.Partials(Base.setindex(zero_tuple(Val(P), T), p, $B - i + b)))
                    v =
                    ∂ =
                    dualv = ForwardDiff.Dual(v,ForwardDiff.Partials(Base.setindex(zero_tuple(Val(P), T), p, $B - i + b)))
                    icc.bandmat[i*$Bp1 + $B - i + b] = dualv
                end
            end
        end
        @inbounds for i ∈ 1:length(icc.x) - $Bp1
            # tᵢ = t[i+$Bp1]
            Base.Cartesian.@nexprs $B b -> begin
                # δt = tᵢ - t[i+b]
                δt = δta[i*$B + $(B*B) + b]
                v1 = absθ_b^(δt-1)
                v = copysign(absθ_b * v1, signθ_b)
                p = δt * v1 * ∂absθ_b
                # v = - exp(θ_b * δt)
                # p = v * δt
                icc.bandmat[i*$Bp1 + $(Bp1*B) + b] = ForwardDiff.Dual(v,ForwardDiff.Partials(Base.setindex(zero_tuple(Val(P), T), p, b)))
            end
        end
        out = - fill_precision_factor_det!(icc.y, icc.bandmat, icc.x)
        Base.Cartesian.@nexprs $P p -> @inbounds icc.∇[p] = out.partials[p]
        out.value
    end
end

@generated function selfdottest(icc::InvCholCovar{T,B,P}, θ::AbstractVector{T}) where {T,B,P}
    Bp1 = B+1
    quote
        Base.Cartesian.@nexprs $B b -> begin
            @inbounds θ_b = θ[b]
            signθ_b = sign(θ_b)
            absθ_b, ∂absθ_b = ∂inv_sigmoid(abs(θ_b))
        end
        δta = icc.δt
        @inbounds  begin
            Base.Cartesian.@nexprs $B i -> begin
                # tᵢ = t[i]
                Base.Cartesian.@nexprs i b -> begin
                    # δt = tᵢ - t[b]
                    δt = δta[i*$B + $B - i + b]
                    v1 = absθ_{$B - i + b}^(δt-1)
                    v = copysign(absθ_{$B - i + b} * v1, signθ_{$B - i + b})
                    p = δt * v1 * ∂absθ_{$B - i + b}
                    # v = - exp(θ_b * δt)
                    # p = v * δt
                    # dual =
                    # @show dual
                    icc.bandmat[i*$Bp1 + $B - i + b] = ForwardDiff.Dual(v,ForwardDiff.Partials(Base.setindex(zero_tuple(Val(P), T), p, $B - i + b)))
                end
            end
        end
        @inbounds for i ∈ 1:length(icc.x) - $Bp1
            # tᵢ = t[i+$Bp1]
            Base.Cartesian.@nexprs $B b -> begin
                # δt = tᵢ - t[i+b]
                δt = δta[i*$B + $(B*B) + b]
                v1 = absθ_b^(δt-1)
                v = copysign(absθ_b * v1, signθ_b)
                p = δt * v1 * ∂absθ_b
                # v = - exp(θ_b * δt)
                # p = v * δt
                icc.bandmat[i*$Bp1 + $(Bp1*B) + b] = ForwardDiff.Dual(v,ForwardDiff.Partials(Base.setindex(zero_tuple(Val(P), T), p, b)))
            end
        end
        logdet = fill_precision_factor_det!(icc.y, icc.bandmat, icc.x)
        out = self_dot(icc.y)
        Base.Cartesian.@nexprs $P p -> @inbounds icc.∇[p] = out.partials[p]
        out.value
    end
end
