randgamma(α::Number) = randgamma(Random.GLOBAL_RNG, α)
randgamma(α::Number, β::Number) = randgamma(Random.GLOBAL_RNG, α, β)
randgamma(rng::AbstractRNG, α::Number, β::Number) = randgamma(rng, α) * β
randgamma(rng::AbstractRNG, α::SIMDPirates.AbstractStructVec) = randgamma(rng, SIMDPirates.extract_data(α))
function randgamma(rng::AbstractRNG, α::Vec{W,T}) where {W,T}
    SVec{W,T}(ntuple(w -> (@inbounds randgamma(rng, α[w].value)), Val(W)))
end
# function randgamma(rng::AbstractRNG, α::T) where T
#     α < 1 ? rand(rng, T)^(1/α) * randgamma_g1(rng, α+1) : randgamma_g1(rng, α)
# end
@inline function randgamma(rng::VectorizedRNG.PCG, α::SVec{W,T}) where {W,T}
    vifelse(α < one(T), SLEEF.exp(-SVec(randexp(rng, Vec{W,T}))/α) * randgamma_g1(rng, α+one(T)), randgamma_g1(rng, α))
end
@inline function randgamma_g1(rng::VectorizedRNG.PCG, α::SVec{W,T}) where {W,T}
    OneThird = vbroadcast(SVec{W,T},one(T)/T(3))
    d = α - OneThird
    c = OneThird * rsqrt(d)
    dv3_out = OneThird
    accepted = SVec(ntuple(i -> Core.VecElement(false), Val(W)))
    while true
        x = SVec(randn(rng, Vec{W,T}))
        v = one(T) + c*x
        mask1 = v > zero(T)
        # v > zero(T) || continue
        v3 = v * v * v
        dv3 = SIMDPirates.evmul(d,v3)
        mask2 = SVec(randexp(rng, Vec{W,T})) > T(-0.5)*x*x - d + dv3 - d*SLEEF.log_fast(v3)
        mask3 = mask1 & mask2
        not_accepted = !accepted
        # not_accepted = SIMDPirates.vnot(accepted)
        dv3_out = vifelse(mask3 & not_accepted, dv3, dv3_out)
        accepted = accepted | mask3
        all(accepted) && return dv3_out
        # all(accepted) && return SIMDPirates.extract_data(dv3_out)
    end
end
@inline randchisq(rng::VectorizedRNG.PCG, ν::SVec{W,T}) where {W,T} = T(2.0) * randgamma(rng, T(0.5)*ν)
@inline randinvchisq(rng::VectorizedRNG.PCG, ν::SVec{W,T}) where {W,T} = T(0.5) / (randgamma(rng, T(0.5)*ν))

@inline function randgamma(rng::AbstractRNG, α::T) where T
    α < one(T) ? exp(-randexp(rng, T)/α) * randgamma_g1(rng, α+one(T)) : randgamma_g1(rng, α)
end
@inline function randgamma_g1(rng::AbstractRNG, α::T) where {T}
    OneThird = one(T)/T(3)
    d = α - OneThird
    @fastmath c = OneThird / sqrt(d)
    @fastmath while true
        x = randn(rng, T)
        v = one(T) + c*x
        v < zero(T) && continue
        v3 = v^3
        dv3 = d*v3
        randexp(rng, T) > T(-0.5)*x^2 - d + dv3 - d*log(v3) && return dv3
    end
end
@inline randchisq(rng::AbstractRNG, ν::T) where {T} = T(2.0) * randgamma(rng, T(0.5)ν)
@inline randchisq(ν::T) where {T} = T(2.0) * randgamma(T(0.5)ν)
@inline randchi(rng::AbstractRNG, ν) = @fastmath sqrt(randchisq(rng, ν))
@inline randchi(ν) = @fastmath sqrt(randchisq(ν))
randdirichlet(α) = randdirichlet(Random.GLOBAL_RNG, α)
### NTuples and SVectors are immutable (we cannot edit them), so we create a new one.
### Note that they are both are stack-allocated, so creating and destroying them
### is fast, does not register as memory allocations, and will never trigger the garbage collector.
### The garbage collector cleans up heap memory, and is slow.
@inline function randdirichlet(rng::AbstractRNG, α)
    γ = randgamma.(Ref(rng), α)
    inv_sum_γ = 1/sum(γ)
    @fastmath typeof(γ).mutable ? γ .*=inv_sum_γ : γ .* inv_sum_γ
end
function randdirichlet!(rng::AbstractRNG, probabilities, α)
    cumulative_γ = zero(eltype(α))
    @inbounds for i ∈ eachindex(α)
        γ = randgamma(rng, α[i])
        # γ = α[i]
        # γ = rand(rng, eltype(α)) * α[i]
        cumulative_γ += γ
        probabilities[i] = γ
    end
    inv_cumulative_γ = 1 / cumulative_γ
    @inbounds for i ∈ eachindex(α)
        probabilities[i] *= inv_cumulative_γ
    end
end
randdirichlet!(probabilities, α) = randdirichlet!(GLOBAL_PCG, probabilities, α)
# @generated function randdirichlet(rng::AbstractRNG, α::NTuple{N,T}) where {T,N}
#     quote
#         $(Expr(:meta,:inline))
#         Base.Cartesian.@nexprs $N n -> γ_n = randgamma(rng, α[n])
#         invsumγ = 1/$(Expr(:call, :+, [Symbol(:γ_,n) for n ∈ 1:N]...))
#         $(Expr(:tuple, [:($(Symbol(:γ_,n))*invsumγ) for n ∈ 1:N]...))
#         # γ_1
#     end
# end
