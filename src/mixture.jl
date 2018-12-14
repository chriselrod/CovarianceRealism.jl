
struct MixtureResults{P,T}
    probs::SVector{P,T}
    scale_factors::SVector{P,T}
end


struct Weight{P,T}
    ω::SVector{P,T}
end
mixture_weights(w::Weight) = exp.(w.ω)
function log_jac(w::Weight)
    sum(w.ω)
end

@inline chi3_pdf(x) = sqrt(x) * exp(-x/2) #/ sqrt(2π)


@generated function (d::SquaredMahalanobisDistances)(x::SVector{TwoPp1,T}) where {TwoPp1,T}
    P, R = divrem(TwoPp1,2)
    R == 1 || throw("Only odd dimensions supported; probability is vector presented as 1 less.")
    quote
        # $(Expr(:meta, :inline))
        π, target = log_jac_probs(Simplex{$(P+1),$T,$P}(SVector{$P,$T}( @ntuple $P p -> x[p]  )))
        untransformed_weights = Weight{$(P+1),$T}(SVector{$(P+1),$T}( @ntuple $(P+1) p -> x[$P+p] ) )
        target -= log_jac(untransformed_weights)
        w = exp.(-2.0 .* untransformed_weights.ω)
        # @show π, w
        # loop = zero(T)
        @nexprs 4 i -> loop_i = zero(T)
        # @show w, π
        N = length(d)
        N4, r = divrem(N, 4)
        # We accumulate 4 seperate sums to increase the accuracy of the summayion (pairwise summation algorithm)
        # If we naively did target +=, we occasionally get convergence failure due to numerical problems.
        # Accumulating the loop on a seperate loop variable seemed to solve the problem.
        # Extending that to pairwise summation, with some number (4, currently) seperate accumulators
        # should further increase accuracy.

        # If we had a random random order, I would accumulate all four in a single loop.
        # But we are likely to sort the Mahalanobis distances, because of use in the PercentileMatching algorthm
        # So, sorting gurantees that each of the loops should be a sum of relatively similarly valued "m"s.
        # This, again, ought to increase accuracy. (summing numbers of different magnitudes is less accurate)
        @nexprs 4 i -> @inbounds for n ∈ 1:N4
            loop_i += log(sum((@ntuple $(P+1) p -> chi3_pdf(d[n+(i-1)*N4]*w[p])*w[p]*π[p] )))
        end
        # Catch remainder of the loop
        @inbounds for n ∈ N-r+1:N
            loop_1 += log(sum((@ntuple $(P+1) p -> chi3_pdf(d[n]*w[p])*w[p]*π[p] )))
        end
        - (target + sum((@ntuple 4 loop))) / 10N
    end
end

@generated function initial_val(v::SVector{P}) where P
    :(vcat((@SVector zeros($(P-1))), v))
end
@generated function extract_values(v::SVector{TwoPp1,T}) where {TwoPp1,T}
    P = TwoPp1 >> 1
    quote
        probs = probabilities(Simplex{$(P+1),$T,$P}(SVector{$P,$T}(@ntuple $P p -> v[p])))
        mixture_sf = mixture_weights(Weight{$(P+1),T}(SVector{$(P+1),$T}(@ntuple $(P+1) p -> v[$P+p])))
        MixtureResults(probs, mixture_sf)
    end
end

function mixture_fit(mahals::SquaredMahalanobisDistances, ::Val{P} = Val(3)) where P
    opt = soptimize(mahals[i], @SVector zeros(2P+1))
    extract_values(opt.minimizer)
end

function mixture_fit!(res::Vector{MixtureResults{P,T}}, mahals::Vector{<:SquaredMahalanobisDistances}) where {P,T}

    for i ∈ eachindex(mahals)
        res[i] = mixture_fit(mahals[i], Val(P))
    end

end


function sample_mixture_distances!(mt::MersenneTwister, d::WeightedSamples, res::MixtureResults{P}) where P
    @fastmath for n ∈ 0:P:length(d)-P
        dist = sqrt(abs2(randn(mt)) + abs2(randn(mt)) + abs2(randn(mt)))
        @inbounds for p ∈ 1:P
            d[n*P + p] = (res.scale_factors[p]*dist, res.probs[p])
        end
    end
end
sample_mixture_distances!(d::WeightedSamples, res::MixtureResults) = sample_mixture_distances!(Random.GLOBAL_RNG, d, res)
