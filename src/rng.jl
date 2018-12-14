randgamma(α) = randgamma(Random.GLOBAL_RNG, α)
randgamma(α, β) = randgamma(Random.GLOBAL_RNG, α, β)
randgamma(mt::MersenneTwister, α, β) = randgamma(mt, α) * β
function randgamma(mt::MersenneTwister, α::T) where T
    d = α - one(T)/3
    @fastmath c = one(T)/sqrt(9d)
    dv3 = zero(T)
    @fastmath while true
        x = randn(mt, T)
        v = one(T) + c*x
        v < zero(T) && continue
        v3 = v^3
        dv3 = d*v3
        randexp(mt, T) > T(-0.5)*x^2 - d + dv3 - d*log(v3) && break
    end
    dv3
end
@inline randchisq(mt::MersenneTwister, ν::T) where T = T(2) * randgamma(mt, T(0.5)ν)
@inline randchisq(ν::T) where T = T(2) * randgamma(T(0.5)ν)
@inline randchi(mt::MersenneTwister, ν) = @fastmath sqrt(randchisq(mt, ν))
@inline randchi(ν) = @fastmath sqrt(randchisq(ν))
randdirichlet(α) = randdirichlet(Random.GLOBAL_RNG, α)
### NTuples and SVectors are immutable (we cannot edit them), so we create a new one.
### Note that they are both are stack-allocated, so creating and destroying them
### is fast, does not register as memory allocations, and will never trigger the garbage collector.
### The garbage collector cleans up heap memory, and is slow.
@inline function randdirichlet(mt::MersenneTwister, α::Union{NTuple,SVector})
    γ = randgamma.(Ref(mt), α)
    γ ./ sum(γ)
end
### If α is not a NTuple, we will assume we can edit it.
@inline function randdirichlet(mt::MersenneTwister, α)
    γ = randgamma.(Ref(mt), α)
    γ ./= sum(γ)
end
