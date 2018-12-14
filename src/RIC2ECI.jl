function RIC2ECI3x3(RIC, r, v)
    y = cross(r, v)
    r̂ = normalize(r)
    ĉ = normalize(y)
    î = cross(ĉ, r̂)
    rotate(RIC, r̂, î, ĉ)
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

    SymmetricM( ECI11, ECI12, ECI22, ECI13, ECI23, ECI33 )
end
