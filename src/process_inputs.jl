
@inline function pdbacksolve(x1,x2,x3,S11,S12,S22,S13,S23,S33)
    @pirate begin
        Ui33 = inv(sqrt(S33))
        U13 = S13 * Ui33
        U23 = S23 * Ui33
        Ui22 = inv(sqrt(S22 - U23*U23))
        U12 = (S12 - U13*U23) * Ui22

        Ui33x3 = SIMDPirates.extract_data(Ui33*x3)

        Ui11 = inv(sqrt(S11 - U12*U12 - U13*U13))
        Ui12 = - U12 * Ui11 * Ui22
        Ui13x3 = - (U13 * Ui11 + U23 * Ui12) * Ui33x3
        Ui23x3 = - U23 * Ui22 * Ui33x3

        (
            Ui11*x1 + Ui12*x2 + Ui13x3,
            Ui22*x2 + Ui23x3,
            Ui33x3
        )
    end
end

@generated function process_big_prop_points!(X::ResizableMatrix{T}, Data::AbstractMatrix{T}) where T
    quote
        resize!(X, size(Data,1))
        @vectorize $T for i ∈ 1:size(Data,1)
            X[i,:] .= pdbacksolve(
                Data[i,1],Data[i,2],Data[i,3],
                Data[i,5],Data[i,6],Data[i,7],Data[i,8],Data[i,9],Data[i,10]
            )
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
    SquaredMahalanobisDistances!(mahals, X)
    nothing
end
