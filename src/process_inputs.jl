
# @inline function pdbacksolve(x1,x2,x3,S11,S12,S22,S13,S23,S33)
#     @pirate begin
#         Ui33 = rsqrt(S33)
#         U13 = S13 * Ui33
#         U23 = S23 * Ui33
#         Ui22 = rsqrt(S22 - U23*U23)
#         U12 = (S12 - U13*U23) * Ui22
#
#         Ui33x3 = SIMDPirates.extract_data(Ui33*x3)
#
#         Ui11 = rsqrt(S11 - U12*U12 - U13*U13)
#         Ui12 = - U12 * Ui11 * Ui22
#         Ui13x3 = - (U13 * Ui11 + U23 * Ui12) * Ui33x3
#         Ui23x3 = - U23 * Ui22 * Ui33x3
#
#         (
#             Ui11*x1 + Ui12*x2 + Ui13x3,
#             Ui22*x2 + Ui23x3,
#             Ui33x3
#         )
#     end
# end
# @inline function pdforwardsolve(x1,x2,x3,S11,S12,S22,S13,S23,S33)
#     R11 = rsqrt(S11)
#     R11x1 = R11 * x1
#     L21 = R11 * S12
#     L31 = R11 * S13
#     R22 = rsqrt(S22 - L21*L21)
#     L32 = R22 * (S23 - L21 * L31)
#     R33 = rsqrt(S33 - L31*L31 - L32*L32)
#
#     nR21x1 = R22 * L21 * R11x1
#     R31x1 = R33 * ( L32*nR21x1 - L31*R11x1 )
#     nR32 = R33 * L32 * R22
#
#     (
#         R11x1,
#         R22*x2 - nR21x1,
#         R31x1 - nR32*x2 + R33*x3
#     )
# end

@generated function process_big_prop_points!(X::ResizableMatrix{T}, BPP::AbstractMatrix{T}) where T
    # quote
    #     # N = size(Data,1)
    #     resize!(X, size(Data,1))
    #     @vectorize $T for i ∈ 1:size(Data,1)
    #         X[i,:] .= pdforwardsolve(
    #             Data[i,1],Data[i,2],Data[i,3],
    #             Data[i,5],Data[i,6],Data[i,7],Data[i,8],Data[i,9],Data[i,10]
    #         )
    #     end
    # end
    quote
        resize!(X, size(BPP,1))
        @vectorize $T for i ∈ 1:size(BPP,1)
            x1 = BPP[i,1]
            x2 = BPP[i,2]
            x3 = BPP[i,3]

            R11 = rsqrt(BPP[i,5])
            R11x1 = SIMDPirates.evmul(R11, x1)
            L21 = R11 * BPP[i,6]
            L31 = R11 * BPP[i,8]
            R22 = rsqrt(BPP[i,7] - L21*L21)
            L32 = R22 * (BPP[i,9] - L21 * L31)
            R33 = rsqrt(BPP[i,10] - L31*L31 - L32*L32)

            nR21x1 = R22 * L21 * R11x1
            R31x1 = R33 * ( L32*nR21x1 - L31*R11x1 )
            nR32 = R33 * L32 * R22

            X[i,1] = R11x1
            X[i,2] = R22*x2 - nR21x1
            X[i,3] = R31x1 - nR32*x2 + R33*x3
        end
    end
end

@generated function process_big_prop_points!(X::AbstractMatrix{T}, BPP::AbstractMatrix{T}) where T
    # quote
    #     @vectorize $T for i ∈ 1:size(Data,1)
    #         X[i,:] .= pdforwardsolve(
    #             Data[i,1],Data[i,2],Data[i,3],
    #             Data[i,5],Data[i,6],Data[i,7],Data[i,8],Data[i,9],Data[i,10]
    #         )
    #     end
    # end
    quote
        @vectorize $T for i ∈ 1:size(BPP,1)
            x1 = BPP[i,1]
            x2 = BPP[i,2]
            x3 = BPP[i,3]

            R11 = rsqrt(BPP[i,5])
            R11x1 = R11 * x1
            L21 = R11 * BPP[i,6]
            L31 = R11 * BPP[i,8]
            R22 = rsqrt(BPP[i,7] - L21*L21)
            L32 = R22 * (BPP[i,9] - L21 * L31)
            R33 = rsqrt(BPP[i,10] - L31*L31 - L32*L32)

            nR21x1 = R22 * L21 * R11x1
            R31x1 = R33 * ( L32*nR21x1 - L31*R11x1 )
            nR32 = R33 * L32 * R22

            X[i,1] = R11x1
            X[i,2] = R22*x2 - nR21x1
            X[i,3] = R31x1 - nR32*x2 + R33*x3
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

# function decorrelate_data!(X, icc::InvCholCovar{T,B}, t) where {T,B}
#     fill_δt!(icc.δt, t, Val(B))
#     resize!(icc, length(t))
#     # @show size(X), size(icc.x)
#     for i ∈ 1:3
#         icc.x .= @view X[:,i]
#         fit!(icc)
#         X[:,i] .= icc.y.data
#     end
# end

function process_BPP!(X, rank1covs, mahals, icc::InvCholCovar, BPP, t)
    process_big_prop_points!(X, BPP)
    decorrelate_data!(icc, X, t)
    generate_rank1covariances!(rank1covs, X)
    MahalanobisDistances!(mahals, X)
    nothing
end
