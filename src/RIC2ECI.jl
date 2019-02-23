function RIC2ECI3x3(RIC, r, v)
    y = LinearAlgebra.cross(r, v)
    r̂ = normalize(r)
    ĉ = normalize(y)
    î = LinearAlgebra.cross(ĉ, r̂)
    rotate(RIC, r̂, î, ĉ)
end

@inline function combinecov(Σ₁, Σ₂, y, z)
    Σ = Σ₁ + Σ₂
    x = LinearAlgebra.cross(y, z)
    a = Σ * x
    SymmetricM2(
        a' * x,
        a' * z,
        quadform(z, Σ)
    )
end

### Is this actually faster than just multiplying matrices???
@inline function rotate(RIC, r̂, î, ĉ)

    @fastmath @inbounds begin
        a = r̂[1]*RIC[1,1] + î[1]*RIC[1,2] + ĉ[1]*RIC[1,3]
        b = r̂[1]*RIC[2,1] + î[1]*RIC[2,2] + ĉ[1]*RIC[2,3]
        c = r̂[1]*RIC[3,1] + î[1]*RIC[3,2] + ĉ[1]*RIC[3,3]

        ECI11 = a * r̂[1] + b * î[1] + c * ĉ[1]
        ECI12 = a * r̂[2] + b * î[2] + c * ĉ[2]
        ECI13 = a * r̂[3] + b * î[3] + c * ĉ[3]


        a = r̂[2]*RIC[1,1] + î[2]*RIC[1,2] + ĉ[2]*RIC[1,3]
        b = r̂[2]*RIC[2,1] + î[2]*RIC[2,2] + ĉ[2]*RIC[2,3]
        c = r̂[2]*RIC[3,1] + î[2]*RIC[3,2] + ĉ[2]*RIC[3,3]

        ECI22 = a * r̂[2] + b * î[2] + c * ĉ[2]
        ECI23 = a * r̂[3] + b * î[3] + c * ĉ[3]

        a = r̂[3]^2*RIC[1,1] + 2r̂[3]*î[3]*RIC[1,2] + 2r̂[3]*ĉ[3]*RIC[1,3]
        b = î[3]^2*RIC[2,2] + 2î[3]*ĉ[3]*RIC[2,3] + ĉ[3]^2*RIC[3,3]

        ECI33 = a + b
    end

    SymmetricM3( ECI11, ECI12, ECI22, ECI13, ECI23, ECI33 )
end

@inline function RIC_to_2D(r₁::SVector{3,T}, v₁::SVector{3,T}, Σ₁::SymmetricM3{T},
                            r₂::SVector{3,T}, v₂::SVector{3,T}, Σ₂::SymmetricM3{T}) where {T}
    δr = r₁ - r₂
    δr₀ = norm(δr)
    δv = v₁ - v₂
    z = LinearAlgebra.cross(δr, δv)

    # ! Relative encounter frame
    nδv = normalize(δv)
    nz  = normalize(z)

    Cp = combinecov(RIC2ECI3x3(Σ₁, r₁, v₁), RIC2ECI3x3(Σ₂, r₂, v₂), nδv, nz)
    Cp, δr₀
end
