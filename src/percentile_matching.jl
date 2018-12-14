
struct PercentileMatch{T} <: AbstractVector{T}
    scale_factors::Vector{T}
end
Base.size(pm::PercentileMatch) = szie(pm.scale_factors)
Base.length(pm::PercentileMatch) = length(pm.scale_factors)
Base.getindex(pm::PercentileMatch, i) = pm.scale_factors[i]
Base.setindex!(pm::PercentileMatch, v, i) = pm.scale_factors[i] = v
Base.resize!(pm::PercentileMatch, N) = resize!(pm.scale_factors, N)

function PercentileMatch(s::SquaredMahalanobisDistances)
    match_percentiles!(PercentileMatch(similar(s.mahal)), s)
end
function PercentileMatch!(pm::PercentileMatch, s::SquaredMahalanobisDistances)
    resize!(pm, length(s))
    match_percentiles!(pm, s)
end

function match_percentiles!(pm::PercentileMatch{T}, s::SquaredMahalanobisDistances{T}) where T
    sort!(s)
    N = length(s)
    denom = T(1) / T(N + 1)
    @inbounds for i ∈ eachindex(s)
        pm[i] = s[i] / quantile(Chisq(3), i * denom)
    end
    pm
end

function sample_scale_factor_distances!(mt::MersenneTwister, d::UniformSamples, res::PercentileMatch)
    @fastmath @inbounds for n ∈ eachindex(d)
        d[n] = sqrt( (abs2(randn(mt)) + abs2(randn(mt)) + abs2(randn(mt))) * rand(mt, res) )
    end
end
sample_scale_factor_distances!(d::UniformSamples, res::PercentileMatch) = sample_scale_factor_distances!(Random.GLOBAL_RNG, d, res)
