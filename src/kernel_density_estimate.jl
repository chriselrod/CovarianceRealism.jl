struct KDE{T,ITP <: ScaledInterpolation} <: ContinuousUnivariateDistribution
    x::StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T}}
    density::Vector{T}
    cumulative_density::Vector{T}
    pdf::ITP
    cdf::ITP
end
function KDE(kde::UnivariateKDE)
    x = kde.x
    density = kde.density
    cumulative_density = cumsum(density)
    cumulative_density ./= cumulative_density[end]

    pdf = Interpolations.scale(interpolate(density, BSpline(Quadratic(Line(OnGrid())))), x)
    cdf = Interpolations.scale(interpolate(cumulative_density, BSpline(Quadratic(Line(OnGrid())))), x)

    KDE(x, density, cumulative_density, pdf, cdf)
end
KDE(distances::AbstractVector) = KDE(kde(distances))
KDE(distances::AbstractVector, weights::AbstractVector) = KDE(kde(distances, weights = weights))
KDE(distances::UniformSamples) = KDE(kde(distances.distances))
KDE(distances::WeightedSamples) = KDE(kde(distances.distances, weights = distances.weights))

# Parameters are summarized by the full (x, density) set
StatsBase.params(kde::KDE) = (kde.x, kde.density)
Statistics.mean(kde::KDE{T}) where T = kde.x' * kde.density * T(kde.x.step)
function Statistics.var(kde::KDE{T}) where T
    μ = zero(T)
    σ² = zero(T)
    x = kde.x; density = kde.density
    @inbounds for i ∈ eachindex(x)
        xd = x[i] * density[i]
        μ += xd
        σ² += x[i] * xd
    end
    (σ² - μ^2) * T(kde.x.step)
end
Statistics.std(kde::KDE) = sqrt(var(kde))
function Statistics.median(kde::KDE)
    kde.x[findfirst(p -> p > 0.5, kde.cumulative_density)]
end
function StatsBase.mode(kde::KDE)
    kde.x[argmax(kde.density)]
end
function StatsBase.entropy(kde::KDE{T}) where T
    out = zero(T)
    density = kde.density
    @inbounds for i ∈ eachindex(density)
        out -= density[i] * log(density[i])
    end
    out * T(kde.x.step)
end
function Distributions.pdf(kde::KDE, x::Real)
    if (x < minimum(kde.x)) || (x > maximum(kde.x))
        p = zero(T)
    else
        p = kde.pdf(x)
    end
end
Distributions.logpdf(kde::KDE, x::Real) = log(kde.pdf(x))
function Distributions.cdf(kde::KDE{T}, x::Real) where T
    if x < minimum(kde.x)
        p = zero(T)
    elseif x > maximum(kde.x)
        p = one(T)
    else
        # the min and max should ideally be unnecessary, but
        # special cases have come up in practice.
        # until we have a better way
        p = min(max(zero(T), kde.cdf(x)),one(T))
    end
    p
end

function Gadfly.layer(kde::KDE, args::Vararg{Union{Function, Gadfly.Element, Theme, Type},N} where N)
    layer(x = kde.x, y = kde.density, Geom.line, args...)
end
