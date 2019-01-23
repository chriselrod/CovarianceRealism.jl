
struct GaussianProcess{T,B} <: DifferentiableObjects.DifferentiableObject{3}
    # Σdata::Vector{T}
    Σ::Symmetric{T,BandedMatrix{T,Array{T,2},Base.OneTo{Int}}}
    ∂Σ::Vector{ForwardDiff.Partials{4,T}}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    t::Vector{T}
    ∇::SizedSIMDArray{Tuple{4},T,1,4,4}
    # ∇::Base.RefValue{ForwardDiff.Partials{4,T}}
end

function GaussianProcess(N::Integer, T = Float64, ::Val{B} = Val{15}()) where B
    # L = N * B - (B*(B+1) >> 1)
    M = B + 1
    data = Matrix{T}(undef, M, N)
    # data = Vector{T}(undef, M * N)
    GaussianProcess{T,B}(
        # data,
        # Symmetric(BandedMatrices._BandedMatrix(reshape(data,(M,N)), Base.OneTo(N), 0, B)),
        Symmetric(BandedMatrices._BandedMatrix(data, Base.OneTo(N), 0, B)),
        # Symmetric(BandedMatrix{T}(undef, N, N, 0, B)),
        Vector{ForwardDiff.Partials{4,T}}(undef, (B+1) * N),
        Vector{T}(undef, N), # x
        Vector{T}(undef, N), # y
        Vector{T}(undef, N), # z
        Vector{T}(undef, N), # t
        # Ref(ForwardDiff.Partials{4,$T}((zero(T),zero(T),zero(T),zero(T))))
        SizedSIMDVector{4,T,4,4}(undef)
    )
end
function GaussianProcess(x::AbstractVector{T}, t::AbstractVector{T}, ::Val{B} = Val{15}()) where {B,T}
    N = length(x)
    # L = N * B - (B*(B+1) >> 1)
    M = B + 1
    data = Matrix{T}(undef, M, N)
    # data = Vector{T}(undef, M * N)
    GaussianProcess{T,B}(
        # data,
        # Symmetric(BandedMatrices._BandedMatrix(reshape(data,(M,N)), Base.OneTo(N), 0, B)),
        Symmetric(BandedMatrices._BandedMatrix(data, Base.OneTo(N), 0, B)),
        Vector{ForwardDiff.Partials{4,T}}(undef, M * N),
        x, # x
        Vector{T}(undef, N), # y
        Vector{T}(undef, N), # z
        t, # t
        # Ref(ForwardDiff.Partials{4,$T}((zero(T),zero(T),zero(T),zero(T))))
        SizedSIMDVector{4,T,4,4}(undef)
    )
end

function Base.resize!(gp::GaussianProcess{T,B}, N::Int) where {T,B}
    M = B + 1
    MN = M * N
    # gp.Σdata = Vector{T}(undef, MN)
    # resize!(gp.Σdata, MN)
    # gp.Σ.data.data = reshape(gp.Σdata, (M, N))
    gp.Σ.data.data = Matrix{T}(undef, (M, N))
    gp.Σ.data.raxis = Base.OneTo(N)
    resize!(gp.∂Σ, MN)
    resize!(gp.x,   N)
    resize!(gp.y,   N)
    resize!(gp.z,   N)
    resize!(gp.t,   N)
    gp
end


# logit(x) = log(x/(1-x))
# inv_logit(x) = 1/(1+exp(-x))
@inline function constrain(θ)
    λₑ = exp(θ[1])
    λᵣ = exp(θ[2])
    αᵣ = exp(θ[3])
    ρ  = inv_logit(θ[4])
    λₑ, λᵣ, αᵣ, ρ
end

function logdettest(θ::AbstractVector{T}, t, ::Val{B} = Val(15)) where {B,T}
    Σ = Symmetric(BandedMatrix{T}(undef, N, N, 0, B))
    data = Σ.data.data
    fill_Σ!(data, t, θ, Val(B))
    @show data[13,15]
    @show data[14,15]
    @show data[15,15]
    2chollogdet!(data, Val(B))
end

function (gp::GaussianProcess{T,B})(θ::AbstractVector{T}) where {T,B}
    Σ = gp.Σ.data.data
    fill_Σ!(gp, θ)
    tri_log_det = 2chollogdet!(Σ, Val(B))
    # @show tri_log_det

    copyto!(gp.y, gp.x)
    BandedMatrices.tbsv!('U', 'T', 'N', size(gp.Σ,1), B, Σ, gp.y)

    tri_log_det + (gp.y' * gp.y) + θ[1] + θ[2]
end
@inline DifferentiableObjects.f(gp::GaussianProcess, θ) = gp(θ)
function DifferentiableObjects.fdf(gp::GaussianProcess{T, B}, θ::AbstractVector{T}) where {T, B}
    λₑ = exp(θ[1])
    λᵣ = exp(θ[2])
    αᵣ = exp(θ[3])
    expnθ₃ = exp(-θ[4])
    ρ = 1 / (1 + expnθ₃)
    ∂ρ = ρ^2 * expnθ₃
    fill_Σ!(gp, λₑ, λᵣ, αᵣ, ρ, ∂ρ)


    # @show gp.∂Σ[13,15]
    # @show gp.∂Σ[14,15]
    # @show gp.∂Σ[15,15]
    # ∇fill_Σ!(gp, θ)

    Σ = gp.Σ.data.data
    tri_log_det = 2chollogdetjac!(Σ, gp.∂Σ, Val(B))
    # @show tri_log_det
    gp.∇ .= tri_log_det.partials
    # gp.∇ .= 0

    # BandedMatrices.p,.btrf!('U', size(gp.Σ,1), B, Σ) # Banded Cholesky
    copyto!(gp.y, gp.x)
    BandedMatrices.tbsv!('U', 'T', 'N', size(gp.Σ,1), B, Σ, gp.y)
    copyto!(gp.z, gp.y)
    BandedMatrices.tbsv!('U', 'N', 'N', size(gp.Σ,1), B, Σ, gp.z)

    calc_gradient!(gp)
    gp.∇[1] += 1
    gp.∇[2] += 1
    tri_log_det.value  + (gp.y' * gp.y) + θ[1] + θ[2]
end
function DifferentiableObjects.scale_fdf(gp::GaussianProcess{T}, θ::AbstractVector{T}, scale_target) where T
    fval = DifferentiableObjects.fdf(gp, θ)
    scale = min(one(T), scale_target / norm(gp.∇))
    SIMDArrays.scale!(gp.∇, scale)
    fval * scale, scale
end
function DifferentiableObjects.scaled_fdf(gp::GaussianProcess{T}, θ::AbstractVector{T}, scale) where T
    fval = DifferentiableObjects.fdf(gp, θ)
    SIMDArrays.scale!(gp.∇, scale)
    fval * scale
end



function calc_gradient!(gp::GaussianProcess{T,B}) where {T,B}
    M = B + 1
    ∇₁, ∇₂, ∇₃, ∇₄ = zero(T), zero(T), zero(T), zero(T)
    ∇ = ForwardDiff.Partials((zero(T),zero(T),zero(T),zero(T)))
    y = gp.y; z = gp.z; ∂Σ = gp.∂Σ
    @inbounds @fastmath for col ∈ 1:min(B,size(∂Σ,2))
        n2zᵢ = -2z[col]
        o = col - B - 1
        for row ∈ B+2-col:M
            Λ = n2zᵢ * y[row + o]
            ∂ = ∂Σ[row + (col-1) * M]
            ∇₁ += Λ * ∂[1]
            ∇₂ += Λ * ∂[2]
            ∇₃ += Λ * ∂[3]
            ∇₄ += Λ * ∂[4]
        end
    end
    @inbounds @fastmath for col ∈ B+1:size(∂Σ,2)
        n2zᵢ = -2z[col]
        o = col - B - 1
        for row ∈ 1:M
            Λ = n2zᵢ * y[row + o]
            ∂ = ∂Σ[row + (col-1) * M]
            ∇₁ += Λ * ∂[1]
            ∇₂ += Λ * ∂[2]
            ∇₃ += Λ * ∂[3]
            ∇₄ += Λ * ∂[4]
        end
    end
    @inbounds begin
        gp.∇[1] += ∇₁
        gp.∇[2] += ∇₂
        gp.∇[3] += ∇₃
        gp.∇[4] += ∇₄
    end
    nothing
end
DifferentiableObjects.gradient(gp::GaussianProcess) = gp.∇

function fill_Σ!(gp::GaussianProcess{T,B}, λₑ, λᵣ, αᵣ, ρ) where {B,T}
    Σ = gp.Σ; t = gp.t
    data = Σ.data.data
    Omρ = 1 - ρ
    @inbounds for col ∈ 1:min(B,size(Σ,2))
        o = col - B - 1
        for row ∈ B+2-col:B
            δ = t[col] - t[row + o] # always positive
            δ² = δ^2
            σₑ = exp(-δ  * λₑ)
            σᵣ = (1 + λᵣ * δ²/(2αᵣ))^(-αᵣ)
            data[row,col] = σₑ * ρ + σᵣ * Omρ
        end
        data[B+1,col] = 1
    end
    @inbounds for col ∈ B+1:size(Σ,2)
        o = col - B - 1
        for row ∈ 1:B
            δ = t[col] - t[row + o] # always positive
            δ² = δ^2
            σₑ = exp(-δ  * λₑ)
            σᵣ = (1 + λᵣ * δ²/(2αᵣ))^(-αᵣ)
            data[row,col] = σₑ * ρ + σᵣ * Omρ
        end
        data[B+1,col] = 1
    end
end
function fill_Σ!(gp::GaussianProcess{T,B}, θ) where {T,B}
    fill_Σ!(gp.Σ.data.data, gp.t, θ, Val(B))
    gp.Σ
end
function fill_Σ!(data, t, θ, ::Val{B}) where {T,B}
    λₑ, λᵣ, αᵣ, ρ = constrain(θ)
    Omρ = 1 - ρ
    @inbounds for col ∈ 1:min(B,size(data,2))
        for row ∈ B+2-col:B
            δ = t[col] - t[col + row - B - 1] # always positive
            δ² = δ^2
            σₑ = exp(-δ  * λₑ)
            σᵣ = (1 + λᵣ * δ²/(2αᵣ))^(-αᵣ)
            data[row,col] = σₑ * ρ + σᵣ * Omρ
        end
        data[B+1,col] = 1
    end
    @inbounds for col ∈ B+1:size(data,2)
        for row ∈ 1:B
            δ = t[col] - t[col + row - B - 1] # always positive
            δ² = δ^2
            σₑ = exp(-δ  * λₑ)
            σᵣ = (1 + λᵣ * δ²/(2αᵣ))^(-αᵣ)
            data[row,col] = σₑ * ρ + σᵣ * Omρ
        end
        data[B+1,col] = 1
    end
end


function fill_Σ!(gp::GaussianProcess{T,B}, λₑ, λᵣ, αᵣ, ρ, ∂ρ) where {B,T}
    M = B + 1
    Σ = gp.Σ; ∂Σ = gp.∂Σ; t = gp.t
    data = Σ.data.data
    Omρ = 1 - ρ
    @inbounds for col ∈ 1:min(B,size(Σ,2))
        for row ∈ B+2-col:B
            δ = t[col] - t[col + row - B - 1] # always positive
            δ² = δ^2
            σₑ = exp(-δ  * λₑ)
            α2 = 2αᵣ
            δ²λᵣ  =δ² * λᵣ
            denom = α2 + δ²λᵣ
            frac = α2 / denom
            σᵣ = frac^αᵣ
            data[row,col] = σₑ * ρ + σᵣ * Omρ
            ∂Σ[row + (col-1) * M] = ForwardDiff.Partials{4,T}((
                    σₑ * ρ * -δ * λₑ,
                    T(-0.5) * σᵣ * δ² * frac * λᵣ * Omρ,
                    σᵣ * (δ²λᵣ + denom * log(frac)) / denom * αᵣ * Omρ,
                    ( σₑ - σᵣ ) * ∂ρ
            ))
        end
        data[B+1,col] = 1
    end
    @inbounds for col ∈ M:size(Σ,2)
        for row ∈ 1:B
            δ = t[col] - t[col + row - B - 1] # always positive
            δ² = δ^2
            σₑ = exp(-δ  * λₑ)
            α2 = 2αᵣ
            δ²λᵣ  = δ² * λᵣ
            denom = α2 + δ²λᵣ
            frac = α2 / denom
            σᵣ = frac^αᵣ
            data[row,col] = σₑ * ρ + σᵣ * Omρ
            ∂Σ[row + (col-1) * M] = ForwardDiff.Partials{4,T}((
                    σₑ * ρ * -δ * λₑ,
                    T(-0.5) * σᵣ * δ² * frac * λᵣ * Omρ,
                    σᵣ * (δ²λᵣ + denom * log(frac)) / denom * αᵣ * Omρ,
                    ( σₑ - σᵣ ) * ∂ρ
            ))
        end
        data[B+1,col] = 1
    end
end


@generated function chol!(U::AbstractArray{T,2}, ::Val{P}) where {T,P}
    M = P+1
    q = quote @fastmath begin end end
    qa = q.args[2].args[3].args
    for p ∈ 1:P
        push!(qa, :(@inbounds Uᵢᵢ = U[$(M + (p-1)*M)]))
        for l ∈ 1:p-1
            j = l - p + M
            push!(qa, :( @inbounds Uⱼᵢ = U[$(l-p+M +  (p-1)*M)] ))
            for k ∈ 1:l-1
                push!(qa, :( @inbounds Uⱼᵢ -= U[$(k-p+M + (p-1)*M)] * U[$(k-l+M + (l-1)*M)] ) )
            end
            push!(qa, :(@inbounds Uⱼᵢ /= U[$(M + (l-1)*M)]))
            push!(qa, :(Uᵢᵢ -= Uⱼᵢ * Uⱼᵢ))
            push!(qa, :(@inbounds U[$(j + (p-1)*M)] = Uⱼᵢ))
        end
        push!(qa, :(@inbounds U[$(M + (p-1)*M)] = sqrt(Uᵢᵢ)))
    end
    loop_expr = quote end
    push!(loop_expr.args, :(Uᵢᵢ = U[$M + i*$M]))
    for p ∈ 1:P
        push!(loop_expr.args, :(l = i + $(p - M) ))
        push!(loop_expr.args, :(@inbounds Uⱼᵢ = U[$p + i * $M]))
        for k ∈ 1:p-1
            push!(loop_expr.args, :(@inbounds Uⱼᵢ -= U[$k + i*$M] * U[$(k-p+M) + l * $M]))
        end
        push!(loop_expr.args, :(@inbounds Uⱼᵢ /= U[$M + l * $M]))
        push!(loop_expr.args, :(Uᵢᵢ -= Uⱼᵢ * Uⱼᵢ))
        push!(loop_expr.args, :(@inbounds U[$p + i*$M] = Uⱼᵢ))
    end
    push!(loop_expr.args, :(@inbounds U[$M + i*$M] = sqrt(Uᵢᵢ)))

    push!(qa, quote
        for i ∈ $P:size(U,2)-1
            $loop_expr
        end
    end)
    q

end

@generated function chollogdet!(U::AbstractArray{T,2}, ::Val{P}) where {T,P}
    M = P+1
    fastmath = true
    if fastmath
        q = quote @fastmath begin logdet = zero($T) end end
        qa = q.args[2].args[3].args
    else
        q = quote logdet = zero($T) end
        qa = q.args
    end
    for p ∈ 1:P
        push!(qa, :(@inbounds Uᵢᵢ = U[$(M + (p-1)*M)]))
        for l ∈ 1:p-1
            j = l - p + M
            push!(qa, :( @inbounds Uⱼᵢ = U[$(l-p+M +  (p-1)*M)] ))
            # push!(qa, :( @show Uⱼᵢ ))
            for k ∈ 1:l-1
                push!(qa, :( @inbounds Uⱼᵢ -= U[$(k-p+M + (p-1)*M)] * U[$(k-l+M + (l-1)*M)] ) )
                # push!(qa, :(@show U[$(k-p+M + (p-1)*M)] ))
                # push!(qa, :(@show U[$(k-l+M + (l-1)*M)] ))
            end
            # push!(qa, :( @show Uⱼᵢ ))
            push!(qa, :(@inbounds Uⱼᵢ /= U[$(M + (l-1)*M)]))
            push!(qa, :(Uᵢᵢ -= Uⱼᵢ * Uⱼᵢ))
            push!(qa, :(@inbounds U[$(j + (p-1)*M)] = Uⱼᵢ))
        end
        push!(qa, :(Uᵢᵢ = sqrt(Uᵢᵢ)))
        push!(qa, :(@inbounds U[$(M + (p-1)*M)] = Uᵢᵢ))
        push!(qa, :(logdet += log(Uᵢᵢ)))
    end
    # push!(qa, :(@show logdet))
    loop_expr = quote end
    push!(loop_expr.args, :(Uᵢᵢ = U[$M + i*$M]))
    for p ∈ 1:P
        push!(loop_expr.args, :(l = i + $(p - M) ))
        push!(loop_expr.args, :(@inbounds Uⱼᵢ = U[$p + i * $M]))
        for k ∈ 1:p-1
            push!(loop_expr.args, :(@inbounds Uⱼᵢ -= U[$k + i*$M] * U[$(k-p+M) + l * $M]))
        end
        push!(loop_expr.args, :(@inbounds Uⱼᵢ /= U[$M + l * $M]))
        push!(loop_expr.args, :(Uᵢᵢ -= Uⱼᵢ * Uⱼᵢ))
        push!(loop_expr.args, :(@inbounds U[$p + i*$M] = Uⱼᵢ))
    end
    push!(loop_expr.args, :(Uᵢᵢ = sqrt(Uᵢᵢ)))
    push!(loop_expr.args, :(@inbounds U[$M + i*$M] = Uᵢᵢ))
    push!(loop_expr.args, :(logdet += log(Uᵢᵢ)))

    push!(qa, quote
        for i ∈ $P:size(U,2)-1
            $loop_expr
        end
        logdet
    end)
    q
end



@generated function chollogdetjac!(U::AbstractArray{T,2}, J::AbstractArray{ForwardDiff.Partials{4,T}}, ::Val{P}) where {T,P}
    M = P+1
    D = ForwardDiff.Dual{nothing,T,4}
    fastmath = true
    if fastmath
        q = quote @fastmath begin logdet = zero(ForwardDiff.Dual{nothing,$T,4}) end end
        qa = q.args[2].args[3].args
    else
        q = quote logdet = zero(ForwardDiff.Dual{nothing,$T,4}) end
        qa = q.args
    end
    for p ∈ 1:P
        push!(qa, :( @inbounds Uᵢᵢ = $D(U[$(M + (p-1)*M)]))) # first access
        for l ∈ 1:p-1
            j = l - p + M
            push!(qa, :( @inbounds Uⱼᵢ = $D(U[$(l-p+M +  (p-1)*M)], J[$(l-p+M +  (p-1)*M)] ) ))
            # push!(qa, :( @show Uⱼᵢ ))
            for k ∈ 1:l-1
                push!(qa, :( @inbounds Uⱼᵢ -= $D(U[$(k-p+M + (p-1)*M)], J[$(k-p+M + (p-1)*M)]) *
                                                $D(U[$(k-l+M + (l-1)*M)], J[$(k-l+M + (l-1)*M)]) ) )
                # push!(qa, :(@show $D(U[$(k-p+M + (p-1)*M)], J[$(k-p+M + (p-1)*P)]) ))
                # push!(qa, :(@show $D(U[$(k-l+M + (l-1)*M)], J[$(k-l+M + (l-1)*P)]) ))
            end
            # push!(qa, :( @show Uⱼᵢ ))
            push!(qa, :(@inbounds Uⱼᵢ /= $D(U[$(M + (l-1)*M)], J[$(M + (l-1)*M)])))
            push!(qa, :(Uᵢᵢ -= Uⱼᵢ * Uⱼᵢ))
            push!(qa, :(@inbounds U[$(j + (p-1)*M)] = Uⱼᵢ.value ))
            push!(qa, :(@inbounds J[$(j + (p-1)*M)] = Uⱼᵢ.partials ))
        end
        push!(qa, :(Uᵢᵢ = sqrt(Uᵢᵢ)))
        push!(qa, :(@inbounds U[$(M + (p-1)*M)] = Uᵢᵢ.value ))
        push!(qa, :(@inbounds J[$(M + (p-1)*M)] = Uᵢᵢ.partials ))
        push!(qa, :(logdet += log(Uᵢᵢ)))
    end
    # push!(qa, :(@show logdet))
    loop_expr = quote end
    push!(loop_expr.args, :(Uᵢᵢ = $D(U[$M + i*$M]))) # first access
    for p ∈ 1:P
        push!(loop_expr.args, :(l = i + $(p - M) ))
        push!(loop_expr.args, :(@inbounds Uⱼᵢ = $D(U[$p + i * $M], J[$p + i * $M])))
        for k ∈ 1:p-1
            push!(loop_expr.args, :(@inbounds Uⱼᵢ -= $D(U[$k + i*$M], J[$k + i*$M]) * $D(U[$(k-p+M) + l * $M], J[$(k-p+M) + l * $M])))
        end
        push!(loop_expr.args, :(@inbounds Uⱼᵢ /= $D(U[$M + l * $M], J[$M + l * $M])))
        push!(loop_expr.args, :(Uᵢᵢ -= Uⱼᵢ * Uⱼᵢ))
        push!(loop_expr.args, :(@inbounds U[$p + i*$M] = Uⱼᵢ.value))
        push!(loop_expr.args, :(@inbounds J[$p + i*$M] = Uⱼᵢ.partials))
    end
    push!(loop_expr.args, :(Uᵢᵢ = sqrt(Uᵢᵢ)))
    push!(loop_expr.args, :(@inbounds U[$M + i*$M] = Uᵢᵢ.value))
    push!(loop_expr.args, :(@inbounds J[$M + i*$M] = Uᵢᵢ.partials))
    push!(loop_expr.args, :(logdet += log(Uᵢᵢ)))

    push!(qa, quote
        for i ∈ $P:size(U,2)-1
            $loop_expr
        end
        logdet
    end)
    q
end

struct GPOpt{T,O}
    state::DifferentiableObjects.BFGSState2{4,T,4,16}
    initial_x::SizedSIMDArray{Tuple{4},T,1,4,4}
    ls::DifferentiableObjects.BackTracking2{O,T,Int}
end

function decorrelate_data!(X::AbstractMatrix{T}, gp::GaussianProcess{T}, gp_opt, t) where T
    N, M = size(X)
    resize!(gp, N)
    gp.t .= t;
    for m ∈ 1:M
        gp.x .= @view X[:,m]
        DifferentiableObjects.optimize_scale!(gp_opt.state, gp, gp_opt.initial_x, gp_opt.ls, T(10), 1e-3)
        X[:,m] .= gp.y
    end
end
