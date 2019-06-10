
@inline function checked_sqrt(x::V) where {W, T, V <: Union{SIMDPirates.Vec{W,T},SIMDPirates.SVec{W,T}}}
    y = SIMDPirates.vsqrt(x)
    SIMDPirates.vifelse(SIMDPirates.visfinite(y), y, SIMDPirates.vbroadcast(V, T(0.005)))
end
@inline function checked_sqrt(x::T) where {T <: Number}
    y = @fastmath sqrt(x)
    isfinite(T) ? y : T(0.005)
end

@generated function process_big_prop_points!(X::ResizableMatrix{T}, BPP::AbstractMatrix{T}) where T
    quote
        resize!(X, size(BPP,1))
        @vectorize $T for i ∈ 1:size(BPP,1)
            y1 = BPP[i,1]
            y2 = BPP[i,2]
            y3 = BPP[i,3]

            R11 = one(T) / checked_sqrt(BPP[i,5])
            # R11 = SIMDPirates.vifelse(SIMDPirates.visfinite(R11)
            x1 = R11 * y1
            L21 = R11 * BPP[i,6]
            L31 = R11 * BPP[i,8]
            R22 = one(T) / checked_sqrt(BPP[i,7] - L21*L21)
            x2 = R22 * (y2 - L21*x1)

            L32 = R22 * (BPP[i,9] - L21 * L31)
            R33 = one(T) / checked_sqrt(BPP[i,10] - L31*L31 - L32*L32)
            
            X[i,1] = x1
            X[i,2] = x2
            X[i,3] = R33 * (y3 - L31*x1 - L32*x2)
        end
    end
end


@generated function process_big_prop_points!(X::AbstractMatrix{T}, BPP::AbstractMatrix{T}) where T
    quote
        @vectorize $T for i ∈ 1:size(BPP,1)
            y1 = BPP[i,1]
            y2 = BPP[i,2]
            y3 = BPP[i,3]

            R11 = one(T) / checked_sqrt(BPP[i,5])
            # R11 = SIMDPirates.vifelse(SIMDPirates.visfinite(R11)
            x1 = R11 * y1
            L21 = R11 * BPP[i,6]
            L31 = R11 * BPP[i,8]
            R22 = one(T) / checked_sqrt(BPP[i,7] - L21*L21)
            x2 = R22 * (y2 - L21*x1)

            L32 = R22 * (BPP[i,9] - L21 * L31)
            R33 = one(T) / checked_sqrt(BPP[i,10] - L31*L31 - L32*L32)
            
            X[i,1] = x1
            X[i,2] = x2
            X[i,3] = R33 * (y3 - L31*x1 - L32*x2)
        end
    end
end

function generate_rank1covariances!(rank1covs::AbstractVector{InverseWishart{T}}, X::AbstractMatrix{T}) where T
    resize!(rank1covs, size(X,1))
    @inbounds for i ∈ 1:size(X,1)
        rank1covs[i] = InverseWishart(X[i,1], X[i,2], X[i,3])
    end
end

function process_BPP!(X, rank1covs, mahals, BPP)
    process_big_prop_points!(X, BPP)
    generate_rank1covariances!(rank1covs, X)
    MahalanobisDistances!(mahals, X)
    nothing
end

function process_BPP!(X, rank1covs, mahals, icc::InvCholCovar, BPP, t)
    process_big_prop_points!(X, BPP)
    decorrelate_data!(icc, X, t)
    generate_rank1covariances!(rank1covs, X)
    MahalanobisDistances!(mahals, X)
    nothing
end
