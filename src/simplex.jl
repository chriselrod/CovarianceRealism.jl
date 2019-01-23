
"""
Stick breaking representation of the simplex.
"""
struct Simplex{P,T,Pm1} <: AbstractVector{T}
    θ::SVector{Pm1,T}
end
Base.length(::Simplex{P}) where P = P
Base.size(::Simplex{P}) where P = (P,)
logit(p) = log(p/(1-p))
inv_logit(y) = 1/(1+exp(-y))

@generated function Simplex(x::NTuple{P,T}) where {P,T}
    q = quote
        y_1 = logit(x[1]) - $(log(1/(P-1)))
        x_sum = zero(T)
    end
    for p ∈ 2:P-1
        push!(q.args, :(x_sum += x[$p]))
        push!(q.args, :( $(Symbol(:y_, p)) = logit(x[$p]/(1-x_sum)) - $(log(1/(P-p))) ) )
    end
    push!(q.args, :(Simplex{$P,$T,$(P-1)}(SVector(@ntuple $(P-1) y))))
    q
end

@generated function Base.getindex(θ::Simplex{P,T,Pm1}, i) where {P,T,Pm1}
    q = quote
        @boundscheck (i > $P || i < 0) && throw(BoundsError())
        x_sum = x = z_1 = inv_logit(θ.θ[1] + $(log(1/(P-1))) )
    end
    for p ∈ 2:min(i,Pm1)
        zsym = Symbol(:z_, p)
        push!(q.args, :( $zsym = inv_logit(θ.θ[$p] + $(log(1/(P-p))) )) )
        push!(q.args, :( x = (1- x_sum)*$zsym ))
        push!(q.args, :( x_sum += x))
    end
    push!(q.args, :( ifelse(i == $P, 1 - x_sum : x) ) )
    q
end
@generated function probabilities(θ::Simplex{P,T,Pm1}) where {P,T,Pm1}
    q = quote
        x_sum = x_1 = z_1 = inv_logit(θ.θ[1] + $(T(log(1/(P-1))) ))
    end
    for p ∈ 2:Pm1
        zsym, xsym = Symbol(:z_, p), Symbol(:x_, p)
        push!(q.args, :( $zsym = inv_logit(θ.θ[$p] + $(T(log(1/(P-p)))) )) )
        push!(q.args, :( $xsym = (1- x_sum)*$zsym ))
        push!(q.args, :( x_sum += $xsym))
    end
    push!(q.args, :( $(Symbol(:x_, P)) = 1 - x_sum ))
    push!(q.args, :( SVector(@ntuple $P p -> x_p) ) )
    q
end
@generated function log_jac(θ::Simplex{P,T,Pm1}) where {P,T,Pm1}
    q = quote
        x_sum = x_1 = z_1 = inv_logit(θ.θ[1] + $(T(log(1/(P-1))) ))
        log_det = log(z_1) + log(1 - z_1)
    end
    for p ∈ 2:Pm1
        zsym, xsym = Symbol(:z_, p), Symbol(:x_, p)
        push!(q.args, :( $zsym = inv_logit(θ.θ[$p] + $(T(log(1/(P-p))) )) ))
        push!(q.args, :( $xsym = (1- x_sum)*$zsym ))
        push!(q.args, :( log_det += log($zsym) + log(1 - $zsym) + log(1 - x_sum) ))
        push!(q.args, :( x_sum += $xsym))
    end
    # push!(q.args, :( $(Symbol(:x_, P)) = 1 - x_sum ))
    push!(q.args, :( SVector(@ntuple $P p -> x_p), log_det ) )
    q
end
@generated function log_jac_probs(θ::Simplex{P,T,Pm1}) where {P,T,Pm1}
    q = quote
        x_sum = x_1 = z_1 = inv_logit(θ.θ[1] + $(T(log(1/(P-1))) ))
        log_det = log(z_1) + log(1 - z_1)
    end
    for p ∈ 2:Pm1
        zsym, xsym = Symbol(:z_, p), Symbol(:x_, p)
        push!(q.args, :( $zsym = inv_logit(θ.θ[$p] + $(T(log(1/(P-p))) )) ))
        push!(q.args, :( $xsym = (1- x_sum)*$zsym ))
        push!(q.args, :( log_det += log($zsym) + log(1 - $zsym) + log(1 - x_sum) ))
        push!(q.args, :( x_sum += $xsym))
    end
    push!(q.args, :( $(Symbol(:x_, P)) = 1 - x_sum ))
    push!(q.args, :( SVector(@ntuple $P p -> x_p), log_det ) )
    q
end
