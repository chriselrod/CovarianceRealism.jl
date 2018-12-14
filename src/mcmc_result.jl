struct MCMCResult{T}
    Probs::Matrix{T}
    RevCholWisharts::Matrix{RevCholWishart{T}}
    CholInvWisharts::Matrix{CholInvWishart{T}}
end
function MCMCResult(NG, iter, ::Type{T} = Float32) where T
    probs = Array{T}(undef, NG, iter)
    revcholwisharts = Array{RevCholWishart{T}}(undef, NG, iter)
    cholinvwisharts = Array{CholInvWishart{T}}(undef, NG, iter)
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
        hcat(a.RevCholWisharts, b.RevCholWisharts),
        hcat(a.CholInvWisharts, b.CholInvWisharts)
    )
end

function Base.cat(a::MCMCResult{T}, b::MCMCResult{T}, iter) where T
    MCMCResult(
        hcat(@view(a.Probs[:,1:iter]),           @view(b.Probs[:,1:iter])),
        hcat(@view(a.RevCholWisharts[:,1:iter]), @view(b.RevCholWisharts[:,1:iter])),
        hcat(@view(a.CholInvWisharts[:,1:iter]), @view(b.CholInvWisharts[:,1:iter]))
    )
end
