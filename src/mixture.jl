# scale factors are standard deviations
# square them for (co)variance
struct MixtureResults{P,T}
    probs::SVector{P,T}
    scale_factors::SVector{P,T}
end

struct MixtureWorkingData{P,T,S,L,O,TI,F}
    state::DifferentiableObjects.BFGSState{P,T,S,L}
    ls::DifferentiableObjects.BackTracking2{O,T,TI}
    obj::F
    initial_x::PaddedMatrices.MutableFixedSizeVector{P,T,S,S}
end

function MixtureWorkingData(mahal::MahalanobisDistances{T}, ::Val{P}, ::Val{BT} = Val(2)) where {T,P,BT}
    initial_x = PaddedMatrices.MutableFixedSizeVector{P,T}(undef)
    MixtureWorkingData(
        DifferentiableObjects.BFGSState(Val(P), T),
        DifferentiableObjects.BackTracking2{T}(Val(BT)),
        OnceDifferentiable(mahal, initial_x),
        initial_x
    )
end

struct Weight{P,T}
    ω::SVector{P,T}
end
mixture_weights(w::Weight) = exp.(w.ω)
function log_jac(w::Weight)
    sum(w.ω)
end

@inline Χ²₃_pdf(x) = sqrt(x) * exp(-x/2) #/ sqrt(2π)
@inline Χ₃_pdf(x) = (x² = x^2; x² * exp(-x²/2)) #/ sqrt(2π)


@generated function (d::MahalanobisDistances)(x::PaddedMatrices.AbstractMutableFixedSizeVector{TwoPp1,T}) where {T,TwoPp1}
    P, R = divrem(TwoPp1,2)
    R == 1 || throw("Only odd dimensions supported; probability is vector presented as 1 less.")
    quote
        @nexprs 4 i -> loop_i = zero(T)
        # $(Expr(:meta, :inline))
        π, target = log_jac_probs(Simplex{$(P+1),$T,$P}(SVector{$P,$T}( @ntuple $P p -> x[p]  )))
        untransformed_weights = Weight{$(P+1),$T}(SVector{$(P+1),$T}( @ntuple $(P+1) p -> x[$P+p] ) )
        target -= log_jac(untransformed_weights)
        w = @. exp( - untransformed_weights.ω)
        # @show π, w
        # loop = zero(T)
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
            loop_i += log(sum((@ntuple $(P+1) p -> Χ₃_pdf(d[n+(i-1)*N4]*w[p])*w[p]*π[p] )))
        end
        # Catch remainder of the loop
        @inbounds for n ∈ N-r+1:N
            loop_1 += log(sum((@ntuple $(P+1) p -> Χ₃_pdf(d[n]*w[p])*w[p]*π[p] )))
        end
        # we normalize by 100N, so that the magnitude scale with sample size
        # The reason for downscaling the magnitude at all is that we optimize this function
        # via the BFGS algorithm. This is a quasi-Newton algorithm.
        # These algorithms are iterative, repeatedly taking steps to find a local minimizer.
        # As a quasi-Newton method, it builds an approximation to the Hessian from changes
        # in the gradient between each of these iterations.
        # Hessians are used to adjust the gradient in determining the next step.
        # However, on the first iteration, the identity matrix is used to approximate the Hessian.
        # Thus, if the gradient is extreme because the magnitude of the function is extreme,
        # it is prone to taking wildly huge steps, landing in poorly conditioned regions of the
        # parameter space, and end up getting stuck and terminating before finding its way back out.
        # (Badly behaved [very non-quadratic] regions can also lead to building poor Hessian approximations,
        #  further slowing down any potential recovery.)
        # By downscaling, we temper the behavior, ensuring more conservative steps before a reasonable
        # Hessian approximation is computed.
        - (target + sum((@ntuple 4 loop))) / ( $(20TwoPp1) * N)
    end
end

@generated function initial_val(v::SVector{P}) where P
    :(vcat((@SVector zeros($(P-1))), v))
end
@generated function extract_values(v::Union{SVector{TwoPp1,T},PaddedMatrices.AbstractFixedSizeVector{TwoPp1,T}}) where {TwoPp1,T}
    P = TwoPp1 >> 1
    quote
        probs = probabilities(Simplex{$(P+1),$T,$P}(SVector{$P,$T}(@ntuple $P p -> v[p])))
        mixture_sf = mixture_weights(Weight{$(P+1),T}(SVector{$(P+1),$T}(@ntuple $(P+1) p -> v[$P+p])))
        MixtureResults(probs, mixture_sf)
    end
end

function mixture_fit(mwd::MixtureWorkingData{TwoPp1, T}, sat_number, prop_point) where {TwoPp1,T}
    for i ∈ 1:20
        opt, scale = DifferentiableObjects.optimize_scale!(mwd.state, mwd.obj, randn!(mwd.initial_x), mwd.ls, 10f0, 1f-5)
        # opt = soptimize(mahals, 0.1 * (@SVector randn(2P+1)))
        all(isfinite, mwd.state.x_old) || continue
        return extract_values(mwd.state.x_old)
    end
    @warn "Mixture failed 20 times. Mixture scale factors `1`. Propogation point: $prop_point, Satellite number: $sat_number."
    extract_values(@SVector zeros(T, TwoPp1))
end

function mixture_fit!(res::Vector{MixtureResults{P,T}}, mahals::Vector{<:MahalanobisDistances}) where {P,T}

    @inbounds for i ∈ eachindex(mahals)
        res[i] = mixture_fit(mahals[i], Val(P))
    end

end


function sample_distances!(rng::AbstractRNG, d::WeightedSamples, res::MixtureResults{P}) where P
    @fastmath for n ∈ 0:P:length(d)-P
        dist = sqrt(abs2(randn(rng)) + abs2(randn(rng)) + abs2(randn(rng)))
        @inbounds for p ∈ 1:P
            d[n + p] = (res.scale_factors[p]*dist, res.probs[p])
        end
    end
    d
end
sample_distances!(d::WeightedSamples, res::MixtureResults) = sample_distances!(Random.GLOBAL_RNG, d, res)

struct MixturePcs{P,T}
    probs::SVector{P,T}
    scale_factors::SVector{P,T}
end

function sample_Pc(rng::AbstractRNG, res::MixtureResults{P,T}, r1, v1, c1, r2, v2, c2, HBR::T2) where {P,T,T2}
    MixturePcs(
        T2.(res.probs),
        SVector{P,T2}(ntuple(p -> pc2dfoster_RIC(r1, v1, c1, r2, v2, c2 * abs2(res.scale_factors[p]), HBR), Val(P)))
    )
end
function calc_Pc(rng, res::AbstractMatrix{T1}, r1, v1, c1, r2, v2, c2, HBR::T2) where {T1,T2}
    T = promote_type(T1,T2)
    Pc = zero(T)
    # column 1 must be of probs
    # column 2 must be of scale factors
    for i ∈ 1:size(res,2)
        p = res[i,1]
        s = res[i,2]
        ### Perhaps do a safer check?
        ### Throw an error, or provide a warning?
        (isfinite(p) && isfinite(s)) || continue
        Pc = muladd(pc2dfoster_RIC(r1,v1,c1,r2,v2,c2 * s^2, HBR), p, Pc)
    end
    Pc
end


function Distributions.logpdf(res::MixtureResults{P,T}, x::SVector{3}) where {P,T}
    probs = res.probs
    scale_factors = res.scale_factors

    m = T(-0.5)*(x' * x)
    ms = zero(T)
    @inbounds @fastmath @simd for p ∈ 1:P
        s = scale_factors[p]
        ms += probs[p] * SLEEFwrap.exp( m / s^2 - T(3)*SLEEFwrap.log(s) )
    end
    # log(ms) - T(1.5log(2π))
    log(ms) - 1.5log(2π)
end

