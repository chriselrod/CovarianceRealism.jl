
"""
The arguments are:

# !    r₁      - Primary object's position vector in ECI coordinates
# !              (1x3 row vector)
# !    v₁      - Primary object's velocity vector in ECI coordinates
# !              (1x3 row vector)
# !    Σ₁      - Primary object's covariance matrix in ECI coordinate frame
# !              (3x3 or 6x6)
# !    r₂      - Secondary object's position vector in ECI coordinates
# !              (1x3 row vector)
# !    v₂      - Secondary object's velocity vector in ECI coordinates
# !              (1x3 row vector)
# !    Σ₂      - Secondary object's covariance matrix in ECI coordinate frame
# !              (3x3 or 6x6)
# !    HBR     - Hard body region

"""
function pc2dfoster_circle(r₁::SVector{3,T}, v₁::SVector{3,T}, Σ₁::SymmetricM3{T},
                            r₂::SVector{3,T}, v₂::SVector{3,T}, Σ₂::SymmetricM3{T}, HBR = T(0.02)) where T
    #
    # ! Construct relative encounter frame
    δr = r₁ - r₂
    δr₀ = norm(δr)
    δv = v₁ - v₂
    z = LinearAlgebra.cross(δr, δv)

    # ! Relative encounter frame
    nδv = normalize(δv)
    nz  = normalize(z)

    Cp = combinecov(Σ₁, Σ₂, nδv, nz)
    Up = revchol(Cp)
    Uv = SVector{3,T}(
        7.071067811865475244008443621048490392848359376884740365883398689953662392310596e-01/Up[1],
        Up[2], Up[3]
    )

    gaussianarea_old(Uv, δr₀, HBR, HBR*HBR)

end
function pc2dfoster_circle_old(r₁::SVector{3,T}, v₁::SVector{3,T}, Σ₁::SymmetricM3{T},
                            r₂::SVector{3,T}, v₂::SVector{3,T}, Σ₂::SymmetricM3{T}, HBR = T(0.02)) where T
    #
    # ! Construct relative encounter frame
    δr = r₁ - r₂
    δr₀ = norm(δr)
    δv = v₁ - v₂
    z = LinearAlgebra.cross(δr, δv)

    # ! Relative encounter frame
    nδv = normalize(δv)
    nz  = normalize(z)

    Cp = combinecov(Σ₁, Σ₂, nδv, nz)
    Up = revchol(Cp)
    Uv = SVector{3,T}(
        7.071067811865475244008443621048490392848359376884740365883398689953662392310596e-01/Up[1],
        Up[2], Up[3]
    )

    gaussianarea_old(Uv, δr₀, HBR, HBR*HBR)

end
@inline function gaussianarea_old(U::SVector{3,T}, x₀::T, HBR = T(0.2), HBR²::T = HBR^2) where T

    n = SVector{8,T}(
        0.09226835946330202,0.273662990072083,0.4457383557765383,
        0.6026346363792564,0.7390089172206591,0.8502171357296142,
        0.9324722294043558,0.9829730996839018
    )
    w = SVector{8,T}(
        0.036704932945846216,0.035454990477090234,0.032997670831211,
        0.029416655081518854,0.02483389042441457,0.01940543741387547,
        0.01331615550646369,0.006773407894247478
    )

    s = HBR / U[3]

    @inbounds Pc = w[1] * gdensity(s*n[1], HBR², x₀, U)
    @inbounds for i ∈ 2:8
        pct = w[i] * gdensity(s*n[i], HBR², x₀, U)
        Pc += pct
    end

    Pc * s

end

function pc2dfoster_RIC(r₁::SVector{3,T}, v₁::SVector{3,T}, Σ₁::SymmetricM3{T},
                            r₂::SVector{3,T}, v₂::SVector{3,T}, Σ₂::SymmetricM3{T}, HBR = T(0.02)) where T
    #
    # ! Construct relative encounter frame
    δr = r₁ - r₂
    δr₀ = norm(δr)
    δv = v₁ - v₂
    z = LinearAlgebra.cross(δr, δv)

    # ! Relative encounter frame
    nδv = normalize(δv)
    nz  = normalize(z)

    Cp = combinecov(RIC2ECI3x3(Σ₁, r₁, v₁), RIC2ECI3x3(Σ₂, r₂, v₂), nδv, nz)
    Up = revchol(Cp)
    Uv = SVector{3,T}(
        7.071067811865475244008443621048490392848359376884740365883398689953662392310596e-01/Up[1],
        Up[2], Up[3]
    )

    gaussianarea_old(Uv, δr₀, HBR, HBR*HBR)

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


#
# @inline function gaussianarea(U::SVector{3,T}, x₀::T, HBR = T(0.2), HBR²::T = HBR^2) where T
#     # ! Relative error within machine error in a test case.
#     # ! This is better than 1e-8 ComputePcUncertainty passed to Pc2dFoster.
#     # ! This algorithm is not adaptive, unlike quadgk.
#     # ! Therefore, it may need validation to ensure that error holds up over a range.
#     # ! Also, currently quadgk is passed AbsTol = 1e-13.
#     # ! With that AbsTol, low probability events guaranteed too much accuracy.
#     # ! quadgk will terminate after its first error estimate
#     # !   - This first error estimate is probably enough for decent relative error
#     # !
#     # !
#     # ! These are Chebyschev nodes and weights of the second kind,
#     # ! with weights adjusted for the implicit weight function.
#     # ! They were generated via the following Julia code:
#     # !
#     # ! #Pkg.add("FastGaussQuadrature") #if it isn't already installed.
#     # ! using FastGaussQuadrature
#     # ! nodes = 32
#     # ! Nh = div(nodes,2)
#     # ! n, w = gausschebyshev(nodes, 2) # Arguments: Number of nodes, Chebyschev-kind (1,2,3,or 4)
#     # ! uw = @. w / sqrt(8pi*(1-abs2(n))) # Counter weight function of Chebyschev's 2nd kind : 1/sqrt(1-x^2)
#     # !
#     # !
#     # ! # Fortran's preprocessor could truncate numbers to single precision if they're not given as double precision.
#     # ! # Fortran also has limits on number of characters per line.
#     # ! # We don't want to have to make these adjustments manually.
#     # ! # The following function converts the vectors into a string that we can then print, copy and paste.
#     # !
#     # ! pa_dp(n[Nh+1:end]) |> print
#     # ! pa_dp(uw[Nh+1:end]) |> print
#     #
#     # ! Note that the rules are symmetric, so I am only copying the positive half.
#     # ! In integrating, the code calculates density for +/- n.
#
#
#     n = SVec{8,T}(
#         0.09226835946330202,0.273662990072083,0.4457383557765383,
#         0.6026346363792564,0.7390089172206591,0.8502171357296142,
#         0.9324722294043558,0.9829730996839018
#     )
#     w = SVec{8,T}(
#         0.036704932945846216,0.035454990477090234,0.032997670831211,
#         0.029416655081518854,0.02483389042441457,0.01940543741387547,
#         0.01331615550646369,0.006773407894247478
#     )
#
#     U₁ = U[1]
#     s = HBR / U[3]
#
#     @fastmath begin
#         z = s * n
#         U₂z = VectorizationBase.extract_data(U[2] * z)
#         U₃z = U[3] * z
#         radical = sqrt( HBR² - U₃z * U₃z )
#         exponentials = exp(-0.5 * z * z)
#         erfs_1 = SLEEFwrap.erfc(erf_args(x₀, U₂z[1].value, U₂z[2].value, radical[1], radical[2], U₁))
#         erfs_2 = SLEEFwrap.erfc(erf_args(x₀, U₂z[3].value, U₂z[4].value, radical[3], radical[4], U₁))
#         erfs_3 = SLEEFwrap.erfc(erf_args(x₀, U₂z[5].value, U₂z[6].value, radical[5], radical[6], U₁))
#         erfs_4 = SLEEFwrap.erfc(erf_args(x₀, U₂z[7].value, U₂z[8].value, radical[7], radical[8], U₁))
#         # erfreductions = SVec{8,Float64}(ntuple(Val(8)) do i
#         #     erfset = erfs[cld(i,2)]
#         #     k = 4*((i-1) % 2)
#         #     VE(erfset[k+1].value + erfset[k+2].value - erfset[k+3].value - erfset[k+4].value)
#         # end)
#         erfreductions = SVec((
#             erfset_reduce(erfs_1, 0), erfset_reduce(erfs_1, 4),
#             erfset_reduce(erfs_2, 0), erfset_reduce(erfs_2, 4),
#             erfset_reduce(erfs_3, 0), erfset_reduce(erfs_3, 4),
#             erfset_reduce(erfs_4, 0), erfset_reduce(erfs_4, 4)
#         ))
#         pc = SVec(VectorizationBase.extract_data(w * exponentials * erfreductions))
#         SIMDPirates.vsum(w * exponentials * erfreductions) * s
#     end
# end
# @inline function erf_args(x₀::T, U₂z₁, U₂z₂, radical₁, radical₂, U₁) where T
#     m₁ = SVec{8,T}(-1, 1,-1, 1,-1, 1,-1, 1)
#     m₂ = SVec{8,T}(-1,-1, 1, 1,-1,-1, 1, 1)
#     U₂z = SVec{8,T}(U₂z₁,U₂z₁,U₂z₁,U₂z₁,U₂z₂,U₂z₂,U₂z₂,U₂z₂)
#     radical = SVec{8,T}(radical₁,radical₁,radical₁,radical₁,radical₂,radical₂,radical₂,radical₂)
#     extract_data(U₁ * muladd(m₂, radical, muladd(m₁, U₂z, x₀)) )
# end
# @inline function erfset_reduce(erfset, k)
#     VE(erfset[k+1].value + erfset[k+2].value - erfset[k+3].value - erfset[k+4].value)
# end

@inline function gdensity(z, HBR², x₀, U)

    @fastmath radical = sqrt( HBR² - (U[3]*z)^2 )

    U₂z = U[2]*z
    # ! Integral over positive and negative half of the ellipse.
    exp(-0.5z^2)*( erfc((x₀-U₂z-radical)*U[1])
                 + erfc((x₀+U₂z-radical)*U[1])
                 - erfc((x₀-U₂z+radical)*U[1])
                 - erfc((x₀+U₂z+radical)*U[1]) )
end

@inline function gdensity(z₁, z₂, HBR², x₀, U)

    @fastmath radical₁ = sqrt( HBR² - (U[3]*z₁)^2 )
    @fastmath radical₂ = sqrt( HBR² - (U[3]*z₂)^2 )

    U₂z₁ = U[2]*z₁
    U₂z₂ = U[2]*z₂

    @fastmath erfcs = SLEEFwrap.erfc(
        VE((x₀-U₂z₁-radical₁)*U[1]),
        VE((x₀+U₂z₁-radical₁)*U[1]),
        VE((x₀-U₂z₁+radical₁)*U[1]),
        VE((x₀+U₂z₁+radical₁)*U[1]),
        VE((x₀-U₂z₂-radical₂)*U[1]),
        VE((x₀+U₂z₂-radical₂)*U[1]),
        VE((x₀-U₂z₂+radical₂)*U[1]),
        VE((x₀+U₂z₂+radical₂)*U[1])
    )

    # ! Integral over positive and negative half of the ellipse.

    nk = SLEEF.exp((VE(-0.5z₁*z₁), VE(-0.5z₂*z₂)))
    erfreduces = (VE( erfcs[1].value
                    + erfcs[2].value
                    - erfcs[3].value
                    - erfcs[4].value ),
                  VE( erfcs[5].value
                    + erfcs[6].value
                    - erfcs[7].value
                    - erfcs[8].value ))
    SIMDPirates.vsum(SIMDPirates.vmul(nk, erfreduces))
end


function pcfoster_core(PriCovECI::SymmetricM3{T}, SecCovECI::SymmetricM3{T},
                        y::SVector{3,T}, z::SVector{3,T}, x₀::T, HBR = T(0.2), HBR² = HBR^2) where T
    Cp = combinecov(PriCovECI, SecCovECI, y, z)
    Up = revchol(Cp)
    Uv = SVector{3,T}(
        7.071067811865475244008443621048490392848359376884740365883398689953662392310596e-01/Up[1],
        Up[2], Up[3]
    )
    gaussianarea(Uv, x₀, HBR, HBR²)
end

function pcfoster_core(Σ::SymmetricM2{T}, x₀::T, HBR = T(0.2), HBR² = HBR^2) where T
    Up = revchol(Σ)
    Uv = SVector{3,T}(
        7.071067811865475244008443621048490392848359376884740365883398689953662392310596e-01/Up[1],
        Up[2], Up[3]
    )
    gaussianarea(Uv, x₀, HBR, HBR²)
end
