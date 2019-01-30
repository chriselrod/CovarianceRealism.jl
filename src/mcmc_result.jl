struct MCMCResult{T}
    Probs::Matrix{T}
    RevCholWisharts::ScatteredMatrix{T,2,RevCholWishart{T}}
    CholInvWisharts::ScatteredMatrix{T,2,CholInvWishart{T}}
end
function MCMCResult(NG, iter, ::Type{T} = Float32) where T
    probs = Array{T}(undef, NG, iter)
    revcholwisharts = ScatteredMatrix{T,2,RevCholWishart{T}}(undef, NG, iter)
    cholinvwisharts = ScatteredMatrix{T,2,CholInvWishart{T}}(undef, NG, iter)
    MCMCResult(probs, revcholwisharts, cholinvwisharts)
end

function MCMCResults(NG, iter, ::Type{T} = Float32, nthread = nthreads()) where T
    mcmc_results = Vector{MCMCResult{T}}(undef, nthread)
    @threads for n ∈ 1:nthread
        mcmc_results[n] = MCMCResult(NG, iter, T)
    end
    mcmc_results
end

function Base.cat(a::MCMCResult{T}, b::MCMCResult{T}) where T
    MCMCResult(
        hcat(a.Probs,           b.Probs),
        ScatteredMatrix{T,2,RevCholWishart{T}}(cat(a.RevCholWisharts.data, b.RevCholWisharts.data, dims = 2)),
        ScatteredMatrix{T,2,CholInvWishart{T}}(cat(a.CholInvWisharts.data, b.CholInvWisharts.data, dims = 2))
    )
end

function Base.cat(a::MCMCResult{T}, b::MCMCResult{T}, iter) where T
    MCMCResult(
        hcat(@view(a.Probs[:,1:iter]),           @view(b.Probs[:,1:iter])),
        ScatteredMatrix{T,2,RevCholWishart{T}}(cat(@view(a.RevCholWisharts[:,1:iter,:]), @view(b.RevCholWisharts[:,1:iter,:]), dims = 2)),
        ScatteredMatrix{T,2,CholInvWishart{T}}(cat(@view(a.CholInvWisharts[:,1:iter,:]), @view(b.CholInvWisharts[:,1:iter,:]), dims = 2))
    )
end


@generated function Random.randn(rng::AbstractRNG, ::Type{SVec{W,T}}) where {W,T}
    :(SVec(@ntuple $W w -> VE(randn(rng, $T))))
end

@generated function sample_distances!(rng::AbstractRNG, wdistances::WeightedSamples{T}, res::MCMCResult{T}) where T
    W = VectorizationBase.pick_vector_width(T)
    V = SVec{W,T}
    quote
        # ciw = res.CholInvWisharts
        copyto!(wdistances.weights, res.Probs)
        distances = wdistances.distances
        ciw = res.CholInvWisharts
        N = length(distances)
        vciw = vectorizable(ciw)
        ptr_distances = vectorizable(distances)
        for n ∈ 1:($W):N+$(1-W)
            vZ = SVector{3}(randn(rng, $V),randn(rng, $V),randn(rng, $V))
            vCIW = vload($V, vciw, n)
            vx = vCIW * vZ
            # @show extract_ν(vCIW)
            u = randchisq(rng, extract_ν(vCIW))
            vstore(sqrt( vx' * vx / u ), ptr_distances, n)
        end
        for n ∈ N+1-(N % $W):N
            z = SVector{3}(randn(rng, T),randn(rng, T),randn(rng, T))
            CIW = ciw[n]
            x = CIW * z
            u = randchisq(rng, extract_ν(CIW))
            distances[n] = sqrt( x' * x / u )
        end
        distances
    end
end

function sample_Pc!(rng::AbstractRNG, wpc_array::WeightedSamples{T1}, res::MCMCResult{T2}, r1, v1, c1, r2, v2, c2, HBR) where {T1,T2}
    rcws = res.RevCholWisharts
    chol_c2 = chol(c2)
    pc_array = wpc_array.distances
    copyto!(wpc_array.weights, res.Probs)
    for i ∈ eachindex(pc_array)
        c2_temp = xtx(randinvwishartfactor(rng, rcws[i]) * chol_c2)
        pc_array[i] = pc2dfoster_RIC(r1, v1, c1, r2, v2, c2_temp, HBR)
    end
    wpc_array
end
