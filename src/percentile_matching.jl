
struct PercentileMatch{T} <: AbstractVector{T}
    scale_factors::Vector{T}
end
Base.size(pm::PercentileMatch) = size(pm.scale_factors)
Base.length(pm::PercentileMatch) = length(pm.scale_factors)
Base.getindex(pm::PercentileMatch, i) = pm.scale_factors[i]
Base.setindex!(pm::PercentileMatch, v, i) = pm.scale_factors[i] = v
Base.resize!(pm::PercentileMatch, N) = resize!(pm.scale_factors, N)

PercentileMatch{T}(undef, N) where T = PercentileMatch(Vector{T}(undef, N))
function PercentileMatch(s::MahalanobisDistances)
    match_percentiles!(PercentileMatch(similar(s.mahal)), s)
end
function PercentileMatch!(pm::PercentileMatch, s::MahalanobisDistances)
    resize!(pm, length(s))
    match_percentiles!(pm, s)
end

function match_percentiles!(pm::PercentileMatch{T}, s::MahalanobisDistances{T}) where T
    sort!(s)
    N = length(s)
    denom = T(1) / T(N + 1)
    @inbounds for i ∈ eachindex(s)
        pm[i] = s[i] / quantile(Chi(3), i * denom)
    end
    pm
end

function sample_distances!(rng::AbstractRNG, d::UniformSamples, res::PercentileMatch)
    @fastmath @inbounds for n ∈ eachindex(d)
        d[n] = sqrt( (abs2(randn(rng)) + abs2(randn(rng)) + abs2(randn(rng))) ) * rand(rng, res)
    end
    d
end
sample_distances!(d::UniformSamples, res::PercentileMatch) = sample_distances!(Random.GLOBAL_RNG, d, res)
