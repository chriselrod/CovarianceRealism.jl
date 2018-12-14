
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
# !    RelTol  - Tolerance used for double integration convergence (usually
# !              set to the value of 1e-08)
# !    HBRType - Type of hard body region. This value needs to be set to one
# !              of the following: 'circle', 'square', 'squareEquArea'
"""
function pc2dfoster_circle(r₁, v₁, Σ₁, r₂, v₂, Σ₂, HBR)
    #
    # ! Construct relative encounter frame
    δr = r₁ - r₂
    δr₀ = norm(δr)
    δv = v₁ - v₂
    z = cross(δr, δv)

    # ! Relative encounter frame
    nδv = normalize(δv)
    nz  = normalize(z)

    Cp = combinecov(Σ₁, Σ₂, nδv, nz)
    Up = revchol(Cp)
    Uv = SVector{3,eltype(Up)}(
        7.071067811865475244008443621048490392848359376884740365883398689953662392310596e-01/Up[1],
        Up[2], Up[3]
    )
    # ! CALCULATE DOUBLE INTEGRAL

    gaussianarea(Uv, δr₀, HBR, HBR²)

end

function combinecov(Σ₁, Σ₂, y, z)
    Σ = Σ₁ + Σ₂
    x = cross(y, z)
    a = Σ * x
    SVector(
        a' * x,
        a' * z,
        quadform(z, Σ)
    )
end



function gaussianarea(U::SVector{3,T}, x₀::T, HBR::T, HBR²::T = HBR^2) where T
    # ! Relative error within machine error in a test case.
    # ! This is better than 1e-8 ComputePcUncertainty passed to Pc2dFoster.
    # ! This algorithm is not adaptive, unlike quadgk.
    # ! Therefore, it may need validation to ensure that error holds up over a range.
    # ! Also, currently quadgk is passed AbsTol = 1e-13.
    # ! With that AbsTol, low probability events guaranteed too much accuracy.
    # ! quadgk will terminate after its first error estimate
    # !   - This first error estimate is probably enough for decent relative error
    # !
    # !
    # ! These are Chebyschev nodes and weights of the second kind,
    # ! with weights adjusted for the implicit weight function.
    # ! They were generated via the following Julia code:
    # !
    # ! #Pkg.add("FastGaussQuadrature") #if it isn't already installed.
    # ! using FastGaussQuadrature
    # ! nodes = 32
    # ! Nh = div(nodes,2)
    # ! n, w = gausschebyshev(nodes, 2) # Arguments: Number of nodes, Chebyschev-kind (1,2,3,or 4)
    # ! uw = @. w / sqrt(8pi*(1-abs2(n))) # Counter weight function of Chebyschev's 2nd kind : 1/sqrt(1-x^2)
    # !
    # !
    # ! # Fortran's preprocessor could truncate numbers to single precision if they're not given as double precision.
    # ! # Fortran also has limits on number of characters per line.
    # ! # We don't want to have to make these adjustments manually.
    # ! # The following function converts the vectors into a string that we can then print, copy and paste.
    # !
    # ! pa_dp(n[Nh+1:end]) |> print
    # ! pa_dp(uw[Nh+1:end]) |> print
    #
    # ! Note that the rules are symmetric, so I am only copying the positive half.
    # ! In integrating, the code calculates density for +/- n.


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

    Pc = w[1] * gdensity(s*n[1], HBR², x₀, U)
    for i ∈ 2:8
        Pc += w[i] * gdensity(s*n[i], HBR², x₀, U)
    end

    Pc * s

end


@inline function gdensity(z, HBR², x₀, U)

    radical = sqrt( HBR² - (U[3]*z)^2 )

    U₂z = U[2]*z
    # ! Integral over positive and negative half of the ellipse.
    exp(-0.5z^2)*( erfc((x₀-U₂z-radical)*U[1])
                 + erfc((x₀+U₂z-radical)*U[1])
                 - erfc((x₀-U₂z+radical)*U[1])
                 - erfc((x₀+U₂z+radical)*U[1]) )
end


function pcfoster_core(PriCovECI, SecCovECI, y, z, x₀, HBR, HBR² = HBR^2)
    Cp = combinecov(PriCovECI, SecCovECI, y, z)
    Up = revchol(Cp)
    Uv = SVector{3,eltype(Up)}(
        7.071067811865475244008443621048490392848359376884740365883398689953662392310596e-01/Up[1],
        Up[2], Up[3]
    )
    gaussianarea(Uv, x₀, HBR, HBR²)
end
