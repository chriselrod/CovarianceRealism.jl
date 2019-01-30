
# @inline function update_probabilities!(rng::AbstractRNG, probabilities, α)
#     probs = randdirichlet(rng, α)
#     @inbounds for i ∈ eachindex(probs)
#         probabilities[i] = probs[i]
#     end
# end
# @inline


function dirichlet!(rng::AbstractRNG, probabilities::AbstractVector{T}, α::NTuple{N,T}) where {N,T}
    cumulative_γ = zero(eltype(α))
    @inbounds for i ∈ eachindex(α)
        γ = randgamma(rng, α[i])
        cumulative_γ += γ
        probabilities[i] = γ
    end
    inv_cumulative_γ = 1 / cumulative_γ
    @inbounds for i ∈ eachindex(α)
        probabilities[i] *= inv_cumulative_γ
    end
end
# @generated function update_probabilities!(rng::AbstractRNG, probabilities, α::NTuple{N,T}) where {N,T}
#     quote
#         # $(Expr(:meta,:inline))
#         Base.Cartesian.@nexprs $N n -> γ_n = @inbounds randgamma(rng, α[n])
#         inv_γ_sum = one($T) / ($(Expr(:call, :+, [Symbol(:γ_, n) for n ∈ 1:N]...)))
#         Base.Cartesian.@nexprs $N n -> @inbounds probabilities[n] = γ_n * inv_γ_sum
#         nothing
#     end
# end

@inline update_probabilities!(probabilities, α) = update_probabilities!(Random.GLOBAL_RNG, probabilities, α)


"""
This function sets the priors of each successive group.
Each Wishart3x3's scale is set to the identity matrix, with 5 degrees of freedom.
The prior probability of group membership is set to 1.5^(number of groups - group number).
"""
@generated function set_priors!(Λs::AbstractVector{InverseWishart{T}}, ::Val{NG}) where {NG, T}
    quote
        $(Expr(:meta, :inline))
        @inbounds @nexprs $NG g ->
            Λs[g] = InverseWishart(one(T), zero(T), zero(T), one(T), zero(T), one(T), T(3), T($(1/NG))*T(1.5)^($NG-g))
        nothing
    end
end

"""
Calculates the Wishart posteriors, conditional on group membership assignments.
It does this via first resetting them to the priors, summing up based on group membership,
and then finally also calculating the corresponding Cholesky factor and inverse.
"""
function calc_Wisharts!(revcholwisharts::AbstractVector{RevCholWishart{T}},
                        cholinvwisharts::AbstractVector{CholInvWishart{T}},
                        invwisharts::AbstractVector{InverseWishart{T}},
                        groups::Groups{NG}, rank1covs::Vector{InverseWishart{T}}) where {T,NG}
    set_priors!(invwisharts, Val(NG))

    for (i,g) ∈ enumerate(groups)
        @inbounds invwisharts[g] += rank1covs[i]
    end

    inv_and_cholesky!(revcholwisharts, cholinvwisharts, invwisharts)
end

# @generated function update_individual_probs!(probabilities::AbstractMatrix{T},
#                                 baseπ::AbstractVector{T}, LiV::AbstractVector{RevCholWishart{T}},
#                                 ν::NTuple{NG,T}, x::AbstractMatrix{T}) where {T,NG}
#     quote
#         @inbounds for g ∈ 1:NG
#             # L is scaled too large by a factor of √ν; cancels out 1/ν factor on quadratic form in t-pdf
#             exponent = T(-0.5) * ν[g] + T(-1.5)
#             Li = LiV[g]
#             Li11             = Li[1]
#             Li21, Li22       = Li[2], Li[4]
#             Li31, Li32, Li33 = Li[3], Li[5], Li[6]
#
#             base = log(baseπ[g]) + log(Li11) + log(Li22) + log(Li33) +
#                     lgamma(-exponent) - lgamma(T(0.5)*ν[g]) - T(1.5)*log(ν[g])
#
#             @cvectorize $T for i ∈ 1:size(x, 1)
#                 lx₁ = Li11*x[i,1]
#                 lx₂ = Li21*x[i,1] + Li22*x[i,2]
#                 lx₃ = Li31*x[i,1] + Li32*x[i,2] + Li33*x[i,3]
#                 probabilities[i,g] = exp(base + exponent * log(one(T) + lx₁*lx₁ + lx₂*lx₂ + lx₃*lx₃))
#             end
#         end
#     end
# end
# @generated function update_individual_probs!(probabilities::AbstractMatrix{T},
#                                 baseπ::AbstractVector{T}, LiV::AbstractVector{RevCholWishart{T}},
#                                 ν::NTuple{NG,T}, x::AbstractMatrix{T}) where {T,NG}
#     quote
#         @inbounds for g ∈ 1:NG
#             # L is scaled too large by a factor of √ν; cancels out 1/ν factor on quadratic form in t-pdf
#             exponent = T(-0.5) * ν[g] + T(-1.5)
#             Li = LiV[g]
#             Li11             = Li[1]
#             Li21, Li22       = Li[2], Li[4]
#             Li31, Li32, Li33 = Li[3], Li[5], Li[6]
#
#             base = log(baseπ[g]) + log(Li11) + log(Li22) + log(Li33) +
#                     lgamma(-exponent) - lgamma(T(0.5)*ν[g]) - T(1.5)*log(ν[g])
#
#             @cvectorize $T for i ∈ 1:size(x, 1)
#                 lx₁ = Li11*x[i,1]
#                 lx₂ = Li21*x[i,1] + Li22*x[i,2]
#                 lx₃ = Li31*x[i,1] + Li32*x[i,2] + Li33*x[i,3]
#                 probabilities[i,g] = log(one(T) + lx₁*lx₁ + lx₂*lx₂ + lx₃*lx₃)
#             end
#             @cvectorize $T for i ∈ 1:size(x, 1)
#                 probabilities[i,g] = exp(base + exponent * probabilities[i,g])
#             end
#         end
#     end
# end

@generated function update_individual_probs!(probabilities::AbstractMatrix{T},
                                baseπ::AbstractVector{T}, LiV::AbstractVector{RevCholWishart{T}},
                                ν::NTuple{NG,T}, x::AbstractMatrix{T}) where {NG,T}

    # unroll is 3 for AVX512f, 1 otherwise
    unroll = max(1, ( VectorizationBase.REGISTER_COUNT - 4) ÷ 9)
    if unroll > 1
        iters = cld(NG, unroll)
         # more evenly distributes iters. Eg, if unroll was 3 and NG 4, now unroll = 2.
        unroll = cld(NG, iters)
    end
    rem = NG % unroll

    q = quote
        base = MVector{$NG, $T}(undef)
        exponent = MVector{$NG, $T}(undef)
        @inbounds for rep ∈ 0:$unroll:$(NG-unroll)
                # L is scaled too large by a factor of √ν; cancels out 1/ν factor on quadratic form in t-pdf
            Base.Cartesian.@nexprs $unroll k -> begin
                exponent_k = T(-0.5) * ν[rep+k] + T(-1.5)
                Li_k = LiV[rep+k]
                Li11_k                 = Li_k[1]
                Li21_k, Li22_k         = Li_k[2], Li_k[4]
                Li31_k, Li32_k, Li33_k = Li_k[3], Li_k[5], Li_k[6]

                exponent[rep+k] = exponent_k
                base[rep+k] = log(baseπ[rep+k]) + log(Li11_k) + log(Li22_k) + log(Li33_k) +
                        lgamma(-exponent_k) - lgamma(T(0.5)*ν[rep+k]) - T(1.5)*log(ν[rep+k])
            end
            @cvectorize $T for i ∈ 1:size(x, 1)
                Base.Cartesian.@nexprs $unroll k -> begin
                    lx₁_k = Li11_k*x[i,1]
                    lx₂_k = Li21_k*x[i,1] + Li22_k*x[i,2]
                    lx₃_k = Li31_k*x[i,1] + Li32_k*x[i,2] + Li33_k*x[i,3]
                    probabilities[i,rep+k] = one(T) + lx₁_k*lx₁_k + lx₂_k*lx₂_k + lx₃_k*lx₃_k
                end
            end
        end
    end
    if rem > 0
        NGmrep = NG - rem
        push!(q.args, quote
                    Base.Cartesian.@nexprs $rem k -> begin
                        exponent_k = T(-0.5) * ν[$NGmrep+k] + T(-1.5)
                        Li_k = LiV[$NGmrep+k]
                        Li11_k                 = Li_k[1]
                        Li21_k, Li22_k         = Li_k[2], Li_k[4]
                        Li31_k, Li32_k, Li33_k = Li_k[3], Li_k[5], Li_k[6]

                        exponent[$NGmrep+k] = exponent_k
                        base[$NGmrep+k] = log(baseπ[$NGmrep+k]) + log(Li11_k) + log(Li22_k) + log(Li33_k) +
                                lgamma(-exponent_k) - lgamma(T(0.5)*ν[$NGmrep+k]) - T(1.5)*log(ν[$NGmrep+k])
                    end
                    @cvectorize $T for i ∈ 1:size(x, 1)
                        Base.Cartesian.@nexprs $rem k -> begin
                            lx₁_k = Li11_k*x[i,1]
                            lx₂_k = Li21_k*x[i,1] + Li22_k*x[i,2]
                            lx₃_k = Li31_k*x[i,1] + Li32_k*x[i,2] + Li33_k*x[i,3]
                            probabilities[i,$NGmrep+k] = one(T) + lx₁_k*lx₁_k + lx₂_k*lx₂_k + lx₃_k*lx₃_k
                        end
                    end
                end)

    end
    push!(q.args, quote
        @cvectorize $T for i ∈ 1:length(probabilities)
            probabilities[i] = log(probabilities[i])
        end
        # @inbounds for g ∈ 1:$NG
        #     @cvectorize $T for i ∈ 1:size(x, 1)
        #         probabilities[i,g] = log(probabilities[i,g])
        #     end
        # endlog
        @inbounds for g ∈ 1:$NG
            expg = exponent[g]
            baseg = base[g]
            @cvectorize $T for i ∈ 1:size(x, 1)
                probabilities[i,g] = baseg + expg * probabilities[i,g]
            end
        end
        @cvectorize $T for i ∈ 1:length(probabilities)
            probabilities[i] = exp(probabilities[i])
        end
    end)
    q
end



function update_individual_probs!(probabilities::AbstractMatrix{T}, baseπ, groups::Groups{NG},
                            Li::AbstractMatrix{T}, ν, x::AbstractMatrix{T}) where {T,NG}
    update_individual_probs!(Random.GLOBAL_RNG, probabilities, baseπ, groups, Li, ν, x)
end


function run_sample!(rng::PCG_Scalar_and_Vector, mcmcres::MCMCResult, workingdata::WorkingData{NG},
                    X::AbstractMatrix{T}, rank1covs::AbstractVector{InverseWishart{T}},
                    baseπ::AbstractVector{T}, chain::Integer = 1, iter = 10000, warmup = 4000) where {T,NG}



    N = length(rank1covs)
    resize!(workingdata, N)

    invwisharts, individual_probs, groups =
        workingdata.inverse_wisharts, workingdata.individual_probs, workingdata.groups

    probs, revcholwisharts, cholinvwisharts =
                    mcmcres.Probs, mcmcres.RevCholWisharts, mcmcres.CholInvWisharts


    # chain jump
    CJ = (chain - 1) * iter + 1

    rand!(rng.vector, groups, baseπ)
    # returning mcmcres keeps revcholwisharts, cholinvwisharts
    # therefore no need for GC.@preserve
    @uviews probs begin
        @inbounds @views begin
            calc_Wisharts!(revcholwisharts[:,CJ], cholinvwisharts[:,CJ], invwisharts, groups, rank1covs)
            for i ∈ 1:warmup
                # @show extract_α(invwisharts, Val(NG))
                randdirichlet!(rng.scalar, probs[:,CJ], extract_α(invwisharts, Val(NG)))
                update_individual_probs!(individual_probs, probs[:,CJ],
                        revcholwisharts[:,CJ], extract_ν(invwisharts, Val(NG)), X)
                rand!(rng.vector, groups, individual_probs)
                calc_Wisharts!(revcholwisharts[:,CJ], cholinvwisharts[:,CJ], invwisharts, groups, rank1covs)
            end
            randdirichlet!(rng.scalar, probs[:,CJ], extract_α(invwisharts, Val(NG)))
            for i ∈ 1:iter-1
                update_individual_probs!(individual_probs, probs[:,CJ+i-1],
                        revcholwisharts[:,CJ+i-1], extract_ν(invwisharts, Val(NG)), X)
                rand!(rng.vector, groups, individual_probs)
                calc_Wisharts!(revcholwisharts[:,CJ+i], cholinvwisharts[:,CJ+i], invwisharts, groups, rank1covs)
                randdirichlet!(rng.scalar, probs[:,CJ+i], extract_α(invwisharts, Val(NG)))
            end
        end
    end
    mcmcres
end
function run_sample!(rng::AbstractRNG, mcmcres::MCMCResult, workingdata::WorkingDataUnifCache{NG},
                    X::AbstractMatrix{T}, rank1covs::AbstractVector{InverseWishart{T}},
                    baseπ::AbstractVector{T}, chain::Integer = 1, iter = 10000, warmup = 4000) where {T,NG}



    N = length(rank1covs)
    resize!(workingdata, N)

    invwisharts, individual_probs, uniform_probs, groups =
        workingdata.inverse_wisharts, workingdata.individual_probs,
        workingdata.uniform_probs, workingdata.groups

    probs, revcholwisharts, cholinvwisharts =
                    mcmcres.Probs, mcmcres.RevCholWisharts, mcmcres.CholInvWisharts


    # chain jump
    CJ = (chain - 1) * iter + 1

    rand!(rng, groups, uniform_probs, baseπ)
    # returning mcmcres keeps revcholwisharts, cholinvwisharts
    # therefore no need for GC.@preserve
    @uviews probs begin
        @inbounds @views begin
            calc_Wisharts!(revcholwisharts[:,CJ], cholinvwisharts[:,CJ], invwisharts, groups, rank1covs)
            for i ∈ 1:warmup
                randdirichlet!(rng, probs[:,CJ], extract_α(invwisharts, Val(NG)))
                update_individual_probs!(individual_probs, probs[:,CJ],
                        revcholwisharts[:,CJ], extract_ν(invwisharts, Val(NG)), X)
                rand!(rng, groups, uniform_probs, individual_probs)
                calc_Wisharts!(revcholwisharts[:,CJ], cholinvwisharts[:,CJ], invwisharts, groups, rank1covs)
            end
            update_probabilities!(rng, probs[:,CJ], extract_α(invwisharts, Val(NG)))
            for i ∈ 1:iter-1
                update_individual_probs!(individual_probs, probs[:,CJ+i-1],
                        revcholwisharts[:,CJ+i-1], extract_ν(invwisharts, Val(NG)), X)
                rand!(rng, groups, uniform_probs, individual_probs)
                calc_Wisharts!(revcholwisharts[:,CJ+i], cholinvwisharts[:,CJ+i], invwisharts, groups, rank1covs)
                randdirichlet!(rng, probs[:,CJ+i], extract_α(invwisharts, Val(NG)))
            end
        end
    end
    mcmcres
end


function singlechain_sample!(mcmcres, X, rank1covs, workingdata, BPP, baseπ, iter = size(mcmcres.Probs, 2), warmup = 4000)

    process_big_prop_points!(X, BPP)
    generate_rank1covariances!(rank1covs, X)

    run_sample!(GLOBAL_PCG, mcmcres, workingdata, X, rank1covs, baseπ, 1, iter, warmup)

end
function singlechain_sample!(rng::AbstractRNG, mcmcres, X, rank1covs, workingdata, BPP, baseπ, iter = size(mcmcres.Probs, 2), warmup = 4000)

    process_big_prop_points!(X, BPP)
    generate_rank1covariances!(rank1covs, X)

    run_sample!(rng, mcmcres, workingdatachains, X, rank1covs, baseπ, 1, iter, warmup)

end

function thread_sample!(mcmcres, X, rank1covs, workingdatachains, BPP, baseπ, iter,
                                        warmup = 4000, chains = length(workingdatachains))

    process_big_prop_points!(X, BPP)
    generate_rank1covariances!(rank1covs, X)

    @threads for chain ∈ 1:chains
        run_sample!(TRNG[chain], mcmcres, workingdatachains[chain], X, rank1covs, baseπ, chain, iter, warmup)
    end

end
