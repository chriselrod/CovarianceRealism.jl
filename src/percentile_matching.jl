# scale factors are standard deviations
# square them for (co)variance
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
    denom = 1 / (N + 1)
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

function sample_Pc!(rng::AbstractRNG, pc_array::AbstractArray, res::AbstractVector, conj, n::Int)
    c1 = SymmetricM3(conj[n,73],conj[n,74],conj[n,79],conj[n,75],conj[n,80],conj[n,84]) / 1e6
    c2 = SymmetricM3(conj[n,133],conj[n,124],conj[n,139],conj[n,135],conj[n,140],conj[n,144]) / 1e6


    r1 = SVector(ntuple(i -> conj[n,i+171], Val(3)))
    v1 = SVector(ntuple(i -> conj[n,i+174], Val(3)))
    r2 = SVector(ntuple(i -> conj[n,i+177], Val(3)))
    v2 = SVector(ntuple(i -> conj[n,i+180], Val(3)))


    HBR = conj[n,157] / 1e3
    sample_Pc!(rng, pc_array, res, r1, v1, c1, r2, v2, c2, HBR)
end


function sample_Pc!(rng::AbstractRNG, pc_array::AbstractArray, res::AbstractVector, r1, v1, c1, r2, v2, c2, HBR)
    resize!(pc_array, length(res))
    @inbounds for i ∈ eachindex(pc_array,res)
        pc_array[i] = pc2dfoster_RIC(r1, v1, c1, r2, v2, c2 * (res[i])^2, HBR)
    end
    pc_array
end

function Distributions.logpdf(res::PercentileMatch{T}, x::SVector{3}) where {T}
    N = length(res)
    m = T(-0.5)*(x' * x)
    ms = zero(T)
    @fastmath @simd for s ∈ res.scale_factors
        ms += SLEEFwrap.exp( m / s^2 - T(3)*SLEEFwrap.log(s) )
    end
    # log(ms) - T(log(N)) - T(1.5log(2π))
    log(ms) - log(N) - 1.5log(2π)
end
