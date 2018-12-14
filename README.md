# CovarianceRealism

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/CovarianceRealism.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/CovarianceRealism.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/CovarianceRealism.jl.svg?branch=master)](https://travis-ci.com/chriselrod/CovarianceRealism.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/CovarianceRealism.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/CovarianceRealism-jl)
[![Codecov](https://codecov.io/gh/chriselrod/CovarianceRealism.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/CovarianceRealism.jl)



***

Sample code utilizing this library:
```julia

using Base.Threads, KissThreading, MAT, CovarianceRealism, HypothesisTests

BPPdata = matread("BigPropPointsFiles.mmm")
# propogation x data x observation
const BigPropPoints = BPPdata["BigPropPoints2"];
const BigSatNumbers = BPPdata["BigSatInfo2"][:,1];

satellites = unique(@view(BigSatNumbers));

# I got an error trying to just "matread" the Conjunctions file
ConjunctionsFile = matopen("Conjunctions.mmm")
const conjunc = read(ConjunctionsFile, "Conjunctions")
close(ConjunctionsFile)

const NTHREADS = nthreads()

T = Float32

const WORKINGDATA__1 = WorkDataChains(N, T, Val( 1), NTHREADS)
const WORKINGDATA__2 = WorkDataChains(N, T, Val( 2), NTHREADS)
const WORKINGDATA__3 = WorkDataChains(N, T, Val( 3), NTHREADS)
const WORKINGDATA__4 = WorkDataChains(N, T, Val( 4), NTHREADS)
const WORKINGDATA__6 = WorkDataChains(N, T, Val( 5), NTHREADS)
const WORKINGDATA__9 = WorkDataChains(N, T, Val( 9), NTHREADS)
const WORKINGDATA_12 = WorkDataChains(N, T, Val(12), NTHREADS)

mcmc_iter = 5000

const MCMCRESULTS__1 = MCMCresults( 1, mcmc_iter, T, NTHREADS)
const MCMCRESULTS__2 = MCMCresults( 2, mcmc_iter, T, NTHREADS)
const MCMCRESULTS__3 = MCMCresults( 3, mcmc_iter, T, NTHREADS)
const MCMCRESULTS__4 = MCMCresults( 4, mcmc_iter, T, NTHREADS)
const MCMCRESULTS__6 = MCMCresults( 6, mcmc_iter, T, NTHREADS)
const MCMCRESULTS__9 = MCMCresults( 9, mcmc_iter, T, NTHREADS)
const MCMCRESULTS_12 = MCMCresults(12, mcmc_iter, T, NTHREADS)

const X_vec = Vector{ResizableMatrix{T}}(undef, NTHREADS)
const rank1cov_vec = Vector{Vector{InverseWishart{T}}}(undef, NTHREADS)
const mahals_vec = Vector{SquaredMahalanobisDistances{T}}(undef, NTHREADS)
const PercentileMatches = Vector{PercentileMatch{T}}(undef, NTHREADS)

@generated function base_π(::Val{N}, ::Type{T}) where {N,T}
    π = SVector(ntuple(n -> T(1 + N - n), Val(N)))
    π /= sum(π)
end

function simulation_iteration(sat_number, prop_to_analyze = 6)
    thread_num = threadid()
    localrng = TRNG]thread_num
    # are satellites guranteed to be sorted? If so, we can probably do this much faster.
    BPP = BigPropPoints[prop_to_analyze, BigSatNumbers .== sat_number, :]
    T = eltype(BPP)

    X = X_vec[thread_num]
    rank1covs = rank1cov_vec[thread_num]
    mahals = mahals_vec[thread_num]

    process_BPP!(X, rank1covs, mahals, BPP)

    mcmc_iter = size(MCMCRESULTS__1.Probs, 2)
    run_sample!(localrng, MCMCRESULTS__1, WORKINGDATA__1,
                    X, rank1covs, base_π(Val( 1), T), 1, mcmc_iter, 500)
    run_sample!(localrng, MCMCRESULTS__2, WORKINGDATA__2,
                    X, rank1covs, base_π(Val( 2), T), 1, mcmc_iter, 500)
    run_sample!(localrng, MCMCRESULTS__3, WORKINGDATA__3,
                    X, rank1covs, base_π(Val( 3), T), 1, mcmc_iter, 500)
    run_sample!(localrng, MCMCRESULTS__4, WORKINGDATA__4,
                    X, rank1covs, base_π(Val( 4), T), 1, mcmc_iter, 500)
    run_sample!(localrng, MCMCRESULTS__6, WORKINGDATA__6,
                    X, rank1covs, base_π(Val( 6), T), 1, mcmc_iter, 500)
    run_sample!(localrng, MCMCRESULTS__9, WORKINGDATA__9,
                    X, rank1covs, base_π(Val( 9), T), 1, mcmc_iter, 500)
    run_sample!(localrng, MCMCRESULTS_12, WORKINGDATA_12,
                    X, rank1covs, base_π(Val(12), T), 1, mcmc_iter, 500)

    pm = PercentileMatches[thread_num]
    PercentileMatch!(pm, mahals)

    mixture_res = mixture_fit(mahals, Val(3))

end

struct SimulationResult{T}

end

sim_res = Vector{SimulationResult{T}}(undef, length(satellites))

tmap!(simulation_iteration, sim_res, satellites)#, batch_size = 10)

# tmap!(f, dst::AbstractArray, src::AbstractArray...; batch_size=1)


S = SymmetricM2(conjunc[1,202], conjunc[1,203], conjunc[1,204]])
```
