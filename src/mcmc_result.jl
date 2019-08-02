struct MCMCResult{T} #<: Distributions.ContinuousUnivariateDistribution
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

@generated function Distributions.logpdf(res::MCMCResult{T}, x::SVector{3}) where {T}
    # W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    # V = SVec{W,T}
    quote
        probs = res.Probs
        rcw = res.RevCholWisharts
        N = length(probs)
        # vrcw = vectorizable(rcw)
        # vprobs = vectorizable(probs)
        p = zero(Float64)
        @vectorize Float64 for n ∈ 1:N
            vRCW = rcw[n]
            vRCWx = vRCW * x
            lkernel = SLEEFwrap.log(1.0 + (vRCWx' * vRCWx))
            ld = logdetw(vRCW)
            ν = extract_ν(vRCW)
            nexponent = 0.5 * ν + 1.5
            base = SLEEFwrap.log(probs[n]) + ld + SLEEFwrap.lgamma(nexponent) - SLEEFwrap.lgamma(0.5ν) - 1.5*SLEEFwrap.log(ν)

            p += SLEEFwrap.exp(base - nexponent * lkernel)
        end
        # log(p) - log($T(size(probs,2))) - $(T(1.5log(π)))
        log(p) - log(size(probs,2)) - $(1.5log(π))
    end
end
@generated function logpdf2(res::MCMCResult{T}, x::SVector{3}) where {T}
    # W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    # V = SVec{W,T}
    quote
        probs = res.Probs
        rcw = res.RevCholWisharts
        N = length(probs)
        # vrcw = vectorizable(rcw)
        # vprobs = vectorizable(probs)
        p = zero(Float64)
        for n ∈ 1:N
            rcw32 = rcw[n]
            vRCW = RevCholWishart{Float64}(Float64.(rcw32.data))
            vRCWx = vRCW * x
            lkernel = log(1.0 + (vRCWx' * vRCWx))
            ld = logdetw(vRCW)
            ν = extract_ν(vRCW)
            nexponent = 0.5 * ν + 1.5
            base = log(Float64(probs[n])) + ld + lgamma(nexponent) - lgamma(0.5ν) - 1.5*log(ν)

            pn = exp(base - nexponent * lkernel)
            # @show pn
            p += pn

        end
        # log(p) - log($T(size(probs,2))) - $(T(1.5log(π)))
        log(p) - log(size(probs,2)) - $(1.5log(π))
    end
end


@generated function Random.randn(rng::AbstractRNG, ::Type{SVec{W,T}}) where {W,T}
    :(SVec(@ntuple $W w -> VE(randn(rng, $T))))
end

@generated function sample_distances!(rng::AbstractRNG, wdistances::WeightedSamples{T}, res::MCMCResult{T}) where {T}
    W = 2*VectorizationBase.pick_vector_width(T)
    V = SVec{W,T}
    quote
        # @show $V
        # ciw = res.CholInvWisharts
        copyto!(wdistances.weights, res.Probs)
        ciw = res.CholInvWisharts
        distances = wdistances.distances
        # ciw = res.CholInvWisharts
        N = length(distances)
        vciw = vectorizable(ciw)
        # @show typeof(ciw), typeof(vciw)
        ptr_distances = vectorizable(distances)
        for n ∈ 1:($W):N+$(1-W)
            vZ = SVector{3}(randn(rng, $V),randn(rng, $V),randn(rng, $V))
            vCIW = vload($V, vciw, n)
            vx = vCIW * vZ
            # @show extract_ν(vCIW)
            u = randchisq(rng, extract_ν(vCIW))
            vstore!(ptr_distances + n - 1, sqrt( vx' * vx / u ))
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

@generated function sample_misses!(table::Array{SVec{W,T},3}, revcholwisharts, probs, r1, v1, r2, v2, c1, c2, N) where {T,W}
    # W = VectorizationBase.pick_vector_width(T)
    quote
        rng = VectorizedRNG.GLOBAL_vPCG
        # Transpose U; broadcast the individual elements.
        chol_c2 = chol(c2)
        for i ∈ eachindex(probs)
            p = vbroadcast(SVec{$W, $T}, probs[i])
            c2_temp = xtx(UpperTriangle3(revcholwisharts[i]) * chol_c2)
            Σ, δr₀ = CovarianceRealism.RIC_to_2D(r1, v1, c1, r2, v2, c2_temp)
            U = chol(Σ)

            L11 = vbroadcast(SVec{$W,$T}, U[1,1])
            L21 = vbroadcast(SVec{$W,$T}, U[1,2])
            L22 = vbroadcast(SVec{$W,$T}, U[2,2])
            vδr₀ = vbroadcast(SVec{$W,$T}, δr₀)
            ν = extract_ν(revcholwisharts[i])
            νinv = 1/ν
            vν = vbroadcast(SVec{$(2W),$T}, (ν < one($T) ? muladd(T(0.5),ν,one($T)) : T(0.5)*ν ))
            @inbounds for n ∈ 1:N
                # Sampling random normals in batches of 4W is fastest on the tested architecture.
                z1_and_2 = randn(rng, Vec{$(4W),$T})

                # We want coordinates z1 and z2. Because we sampled 4W, we have 2W of each z1 and z2.
                # In some versions of Julia there is a bug that causes a more typical version of constructing these tuples:
                # z1_1 = ntuple(Val($W)) do i x1_and_2[i] end
                # or equivalently
                # z1_1 = ntuple(i -> x1_and_2[i], Val($W))
                # So we construct the tuple explicitly here. This is fast in all versions of Julia > 1.
                z1_1 = SVec($(Expr(:tuple, [:(z1_and_2[$w]) for w ∈    1: W]...)))
                z1_2 = SVec($(Expr(:tuple, [:(z1_and_2[$w]) for w ∈  W+1:2W]...)))
                z2_1 = SVec($(Expr(:tuple, [:(z1_and_2[$w]) for w ∈ 2W+1:3W]...)))
                z2_2 = SVec($(Expr(:tuple, [:(z1_and_2[$w]) for w ∈ 3W+1:4W]...)))


                u1_and_2 = vbroadcast(SVec{$(2W),$T}, T(0.5)) /
                    (
                        ν < one($T) ?
                            SIMDPirates.evmul(SLEEF.exp(-SVec(randexp(rng, Vec{$(2W),$T}))*νinv), randgamma_g1(rng, vν)) :
                            randgamma_g1(rng, vν)
                    ) |> SIMDPirates.extract_data

                u1 = sqrt(SVec($(Expr(:tuple, [:(u1_and_2[$w]) for w ∈   1: W]...))))
                u2 = sqrt(SVec($(Expr(:tuple, [:(u1_and_2[$w]) for w ∈ W+1:2W]...))))
                # Multiply Calculate L * z = x.
                # Additrionally, scaling by u, the random chi square, to sample from the multivariate t distribution.
                # Additionally, subtract the distance δr₀, which we treat as [δr₀, 0]
                x1_1 = u1 * L11 * z1_1 - vδr₀
                x1_2 = u2 * L11 * z1_2 - vδr₀

                x2_1 = u1 * (L21 * z1_1 + L22 * z2_1)
                x2_2 = u2 * (L21 * z1_2 + L22 * z2_2)


                # Calculate distance, multiply 1000 to translate RevCholWishartsfrom kilometers to meters.
                δ_1 = SIMDPirates.evmul(sqrt( x1_1*x1_1 + x2_1*x2_1 ), T(10^3))
                δ_2 = SIMDPirates.evmul(sqrt( x1_2*x1_2 + x2_2*x2_2 ), T(10^3))
                vi = vbroadcast(SVec{$W,$T}, $T(100))
                l_1_5 = δ_1 < vi
                l_2_5 = δ_2 < vi
                # Every 5 times, we check if we can break early.
                for i ∈ 50:-5:1
                    ((l_1_5 === SVec($(Expr(:tuple, [:(Core.VecElement(false)) for w ∈ 1:W]...)))) && (l_2_5 === SVec($(Expr(:tuple, [:(Core.VecElement(false)) for w ∈ 1:W]...))))) && break
                    # (any(l_1_5) || any(l_2_5)) || break
                    Base.Cartesian.@nexprs 5 j -> begin
                        vi_j = vbroadcast(SVec{$W,$T}, $T(i+1-j))
                        ti_1_j = table[1,j,i+1-j]
                        l_1_j = δ_1 < vi_j
                        ti_1_j = vifelse(l_1_j, ti_1_j + p, ti_1_j)
                        table[1,j,i+1-j] = ti_1_j
                        ti_2_j = table[2,j,i+1-j]
                        l_2_j = δ_2 < vi_j
                        ti_2_j = vifelse(l_2_j, ti_2_j + p, ti_2_j)
                        table[2,j,i+1-j] = ti_2_j
                    end
                end
            end

        end

    end
end



function sample_Pc!(rng::AbstractRNG, wpc_array::WeightedSamples{T1}, res::MCMCResult{T2}, r1, v1, c1, r2, v2, c2, HBR) where {T1,T2}
    sample_Pc!(rng, wpc_array, res.RevCholWisharts, res.Probs, r1, v1, c1, r2, v2, c2, HBR)
end

function sample_Pc!(rng::AbstractRNG, wpc_array::WeightedSamples{T1}, rcws::ScatteredMatrix{T2,2,RevCholWishart{T2}}, probs, r1, v1, c1, r2, v2, c2, HBR) where {T1,T2}
    chol_c2 = chol(c2)
    pc_array = wpc_array.distances
    copyto!(wpc_array.weights, probs)
    for i ∈ eachindex(pc_array)
        rcws64 = RevCholWishart{Float64}(Float64.(rcws[i].data))
        c2_temp = xtx(randinvwishartfactor(rng, rcws64) * chol_c2)
        pc_array[i] = pc2dfoster_RIC(r1, v1, c1, r2, v2, c2_temp, HBR)
    end
    wpc_array
end
function sample_Pc!(rng::AbstractRNG, pc_array::AbstractVector{T1}, rcws::ScatteredMatrix{T2,2,RevCholWishart{T2}}, r1, v1, c1, r2, v2, c2, HBR) where {T1,T2}
    chol_c2 = chol(c2)
    for i ∈ eachindex(pc_array)
        rcws64 = RevCholWishart{Float64}(Float64.(rcws[i].data))
        c2_temp = xtx(randinvwishartfactor(rng, rcws64) * chol_c2)
        pc_array[i] = pc2dfoster_RIC(r1, v1, c1, r2, v2, c2_temp, HBR)
    end
    pc_array
end
# This method also calculates covariance ratios
function sample_Pc!(rng::AbstractRNG, pc_array::AbstractMatrix{T1}, rcws::ScatteredMatrix{T2,2,RevCholWishart{T2}}, r1, v1, c1, r2, v2, c2, HBR) where {T1,T2}
    chol_c2 = chol(c2)
    ldc = logdet(chol_c2)
#    @show c2
    for i ∈ 1:size(pc_array,1)
        # RevCholWishart is precision factor
        rcws64 = RevCholWishart{Float64}(Float64.(rcws[i].data))
        rw = randinvwishartfactor(rng, rcws64)
        cov_factor = rw * chol_c2
        c11 = cov_factor[1,1]
        c22 = cov_factor[2,2]
        c33 = cov_factor[3,3]
        if !((c11 > 0) && (c22 > 0) && (c33 > 0))
            @show c11, c22, c33
            @show rcws64
            @show rw
            @show chol_c2
            @show cov_factor
        end
        
        pc_array[i,2] = log(cov_factor[1,1]) + log(cov_factor[2,2]) + log(cov_factor[3,3]) - ldc
#        pc_array[i,2] = logdet(cov_factor) - ldc
        c2_temp = xtx(cov_factor)
#        if i <= 4
#            @show rcws64.data
#            @show rw
#            @show c2_temp
#        end
        pc_array[i,1] = pc2dfoster_RIC(r1, v1, c1, r2, v2, c2_temp, HBR)
    end
    pc_array
end
