
struct Groups{NG} <: AbstractVector{Int8}
    groups::Vector{Int8}
    Groups{NG}(::UndefInitializer, N) where NG = new(Vector{Int8}(undef, N))
end
Base.length(g::Groups{NG}) where NG = length(g.groups)
Base.size(g::Groups{NG}) where NG = size(g.groups)
@inline Base.getindex(g::Groups, i) = g.groups[i]
@inline Base.setindex!(g::Groups, v, i) = g.groups[i] = v
@inline Base.IndexStyle(::Groups) = IndexLinear()
@inline Base.pointer(g::Groups) = pointer(g.groups)
@inline Base.resize!(g::Groups, i::Int) = resize!(g.groups, i)

# Should these functions be located here, or in rng.jl?

@generated function Random.rand!(rng::AbstractRNG, group::Groups{NG}, unifs::Vector{T},
                                                    probabilities::AbstractMatrix{T}) where {NG,T}
    W = SLEEFwrap.pick_vector_width(T)
    Vi = Vec{W,Int8}
    Vf = Vec{W,T}
    p_NG = Symbol(:p_, NG)
    vp_NG = Symbol(:vp_, NG)
    vg_NG = Symbol(:vg_, NG)
    quote
        N = length(group)
        rand!(rng, unifs)
        @nexprs $NG g -> vg_g = vbroadcast($Vi, g)
        pg = pointer(group)
        for n ∈ 1:($W):N+$(1-W)
            vp_1 = vload($Vf, probabilities, n)
            @nexprs $(NG-1) g -> vp_{g+1} = SIMDPirates.vadd(vp_g, vload($Vf, probabilities, (n,g+1)))
            vu = SIMDPirates.vmul(vload($Vf, unifs, n), $vp_NG)
            vg = $vg_NG
            @nexprs $(NG-1) g -> vg = vifelse( SIMDPirates.vless(vu, vp_{$NG-g}), vg_{$NG-g}, vg )
            vstore(vg, pg - 1 + n)
        end
        for n ∈ N+1-(N % $W):N
            p_1 = probabilities[n]
            @nexprs $(NG-1) g -> p_{g+1} = p_g + probabilities[n,g+1]
            ptup = @ntuple $NG p
            u = unifs[n] * $p_NG
            g = Int8(1)
            while u > ptup[g]
                g += Int8(1)
            end
            group[n] = g
        end
    end
end


@generated function Random.rand!(rng::AbstractRNG, group::Groups{NG}, unifs::AbstractVector{T},
                                                    probabilities::AbstractVector{T}) where {NG,T}
    W = SLEEFwrap.pick_vector_width(T)
    Vi = Vec{W,Int8}
    Vf = Vec{W,T}
    p_NG = Symbol(:p_, NG)
    vp_NG = Symbol(:vp_, NG)
    vg_NG = Symbol(:vg_, NG)
    quote
        N = length(group)
        rand!(rng, unifs)
        @nexprs $NG g -> vg_g = vbroadcast($Vi, g)

        p_1 = probabilities[1]
        @nexprs $(NG-1) g -> p_{g+1} = p_g + probabilities[g+1]
        @nexprs $NG g -> vp_g = vbroadcast($Vf, p_g)
        pg = pointer(group)
        for n ∈ 0:($W):N-$W
            vu = SIMDPirates.vmul(vload($Vf, unifs, n), $vp_NG)
            vg = $vg_NG
            @nexprs $(NG-1) g -> vg = vifelse( SIMDPirates.vless(vu, vp_{$NG-g}), vg_{$NG-g}, vg )
            vstore(vg, pg + n)
        end
        r = N % $W
        if r > 0
            ptup = @ntuple $NG p
            for n ∈ N+1-r:N
                u = unifs[n] * $p_NG
                g = Int8(1)
                while u > ptup[g]
                    g += Int8(1)
                end
                group[n] = g
            end
        end
    end
end


Random.rand!(g::Groups, α::AbstractArray) = Random.rand!(Random.GLOBAL_RNG, g, similar(α, size(α,1)), α)
function Random.rand!(g::Groups, unifs::AbstractVector, α::AbstractArray)
        Random.rand!(Random.GLOBAL_RNG, g, unifs, α)
end
Random.rand!(g::Groups) = Random.rand!(Random.GLOBAL_RNG, g)
function Random.rand!(rng::AbstractRNG, g::Groups{NG}) where NG
    @inbounds for n ∈ eachindex(g.groups)
        g[n] = rand(rng, Int8(1):Int8(NG))
    end
end
function Random.rand(::Type{Groups{NG}}, α::AbstractMatrix) where NG
    Random.rand!(Random.GLOBAL_RNG, Groups{NG}(undef,size(α,1)), α)
end
function Random.rand(rng::AbstractRNG, ::Type{Groups{NG}}, α::AbstractMatrix) where NG
    Random.rand!(rng, Groups{NG}(undef,size(α,1)), α)
end
function Random.rand(::Type{Groups{NG}}, N::Integer) where NG
    g = Groups{NG}(undef, N)
    rand!(g)
    g
end
