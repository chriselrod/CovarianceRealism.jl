
@inline function update_probabilities!(mt::MersenneTwister, probabilities, α)
    @inbounds probabilities .= randdirichlet(mt, α)
    nothing
end
@inline update_probabilities!(probabilities, α) = update_probabilities!(Random.GLOBAL_RNG, probabilities, α)



@generated function set_priors!(Λs::AbstractVector{InverseWishart{T}}, ::Val{NG}) where {NG, T}
    quote
        $(Expr(:meta, :inline))
        @inbounds @nexprs $NG g ->
            Λs[g] = InverseWishart(one(T), zero(T), zero(T), one(T), zero(T), one(T), T(3), T(1.5)^($NG-g))
        nothing
    end
end


function calc_Wisharts!(revcholwisharts::AbstractVector{RevCholWishart{T}},
                        cholinvwisharts::AbstractVector{CholInvWishart{T}},
                        invwisharts::AbstractVector{InverseWishart{T}},
                        groups::Groups{NG}, rank1covs::Vector{InverseWishart{T}}) where {T,NG}
    set_priors!(invwisharts, Val(NG))

    @inbounds for (i,g) ∈ enumerate(groups)
        invwisharts[g] += rank1covs[i]
    end

    inv_and_cholesky!(revcholwisharts, cholinvwisharts, invwisharts)
end

@generated function update_individual_probs!(mt::MersenneTwister, probabilities::AbstractMatrix{T},
                                baseπ::AbstractVector{T}, LiV::AbstractVector{RevCholWishart{T}},
                                ν::NTuple{NG,T}, x::AbstractMatrix{T}) where {T,NG}
    quote
        @inbounds for g ∈ 1:NG
            # L is scaled too large by a factor of √ν; cancels out 1/ν factor on quadratic form in t-pdf
            Li = LiV[g]
            Li11             = Li[1]
            Li21, Li22       = Li[2], Li[4]
            Li31, Li32, Li33 = Li[3], Li[5], Li[6]

            exponent = T(-0.5) * ν[g] + T(-1.5)
            base = log(baseπ[g]) + log(Li11) + log(Li22) + log(Li33) +
                    lgamma(-exponent) - lgamma(T(0.5)*ν[g]) - T(1.5)*log(ν[g])

            @vectorize $T for i ∈ 1:size(x, 1)
                lx₁ = Li11*x[i,1]
                lx₂ = Li21*x[i,1] + Li22*x[i,2]
                lx₃ = Li31*x[i,1] + Li32*x[i,2] + Li33*x[i,3]
                probabilities[i,g] = exp(base + exponent * log(one(T) + lx₁*lx₁ + lx₂*lx₂ + lx₃*lx₃))
            end
        end
    end
end
function update_individual_probs!(probabilities::AbstractMatrix{T}, baseπ, groups::Groups{NG},
                                    Li::AbstractMatrix{T}, ν, x::AbstractMatrix{T}) where {T,NG}
    update_individual_probs!(Random.GLOBAL_RNG, probabilities, baseπ, groups, Li, ν, x)
end


function run_sample!(mt::MersenneTwister, mcmcres::MCMCResult, workingdata::WorkingData{NG},
                    X::AbstractMatrix{T}, rank1covs::AbstractVector{InverseWishart{T}},
                    baseπ::AbstractVector{T}, chain::Integer = 1, iter = 10000, warmup = 4000) where {T,NG}




    invwisharts, individual_probs, uniform_probs, groups =
        workingdata.inverse_wisharts, workingdata.individual_probs,
        workingdata.uniform_probs, workingdata.groups

    probs, revcholwisharts, cholinvwisharts =
                    mcmcres.Probs, mcmcres.RevCholWisharts, mcmcres.CholInvWisharts

    resize!(groups, size(X,1))
    # chain jump
    CJ = (chain - 1) * iter + 1

    N = length(rank1covs)
    rand!(mt, groups, uniform_probs, baseπ)
    @uviews probs revcholwisharts cholinvwisharts begin
        @inbounds @views begin
            calc_Wisharts!(revcholwisharts[:,CJ], cholinvwisharts[:,CJ], invwisharts, groups, rank1covs)
            for i ∈ 1:warmup
                update_probabilities!(mt, probs[:,CJ], extract_α(invwisharts, Val(NG)))
                update_individual_probs!(mt, individual_probs, probs[:,CJ],
                        revcholwisharts[:,CJ], extract_ν(invwisharts, Val(NG)), X)
                rand!(mt, groups, uniform_probs, individual_probs)
                calc_Wisharts!(revcholwisharts[:,CJ], cholinvwisharts[:,CJ], invwisharts, groups, rank1covs)
            end
            update_probabilities!(mt, probs[:,CJ], extract_α(invwisharts, Val(NG)))
            for i ∈ 1:iter-1
                update_individual_probs!(mt, individual_probs, probs[:,CJ+i-1],
                        revcholwisharts[:,CJ+i-1], extract_ν(invwisharts, Val(NG)), X)
                rand!(mt, groups, uniform_probs, individual_probs)
                calc_Wisharts!(revcholwisharts[:,CJ+i], cholinvwisharts[:,CJ+i], invwisharts, groups, rank1covs)
                update_probabilities!(mt, probs[:,CJ+i], extract_α(invwisharts, Val(NG)))
            end
        end
    end
    mcmcres
end



function thread_sample!(mcmcres, X, rank1covs, workingdatachains, BPP, baseπ, iter,
                                        warmup = 4000, chains = length(workingdatachains))

    process_big_prop_points!(X, BPP)
    generate_rank1covariances!(rank1covs, X)

    @threads for chain ∈ 1:chains
        run_sample!(TRNG[chain], mcmcres, workingdatachains[chain], X, rank1covs, baseπ, chain, iter, warmup)
    end

end
