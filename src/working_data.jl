

struct WorkingData{NG,T}
    inverse_wisharts::Vector{InverseWishart{T}}
    individual_probs::ResizableMatrix{T,NG}
    uniform_probs::Vector{T}
    groups::Groups{NG}
end
function WorkingData(N, ::Type{T}, ::Val{NG}) where {T,NG}
    WorkingData(
        Vector{InverseWishart{T}}(undef, NG), # inverse_wisharts
        ResizableMatrix{T,NG}(undef, N), # individual_probs
        Vector{T}(undef, N), #uniform_probs
        Groups{NG}(undef, N) # groups
    )
end
function Base.resize!(wd::WorkingData, N)
    resize!(wd.individual_probs, N)
    resize!(wd.uniform_probs, N)
    resize!(wd.groups, N)
    wd
end

function WorkDataChains(N, ::Type{T}, ::Val{NG}, chains) where {T,NG}
    workdatachains = Vector{WorkingData{NG,T}}(undef, chains)
    @threads for i ∈ 1:chains
        workdatachains[i] = WorkingData(N, T, Val(NG))
    end
    workdatachains
end
