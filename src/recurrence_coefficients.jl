export rc_legendre01


"""
This script contains routines for building non-monic orthogonal polynomials.
"""

"""
    rc_legendre01(N::Int, normed=false)

Creates `N` recurrence coefficents for non-monic Legendre Polynomials 
that are orthogonal on ``(0, 1)`` relative to ``w(x) = (1)`` and follow:

``∫ w(x)fn(x)^2dx = hn`` where hn = (1 / 2n + 1) for n = 0, 1, 2...``

The three term recurrence relation is given by:
``P₍ₙ₊₁₎(x) = αₙPₙ(x) - βₙP₍ₙ₋₁₎(x)``

where:
``αₙ = (2n + 1) / (n + 1)``

``βₙ = n / (n + 1)``

``P₀(x) = 1`` and 

``P₋₁(x) = 0``


If `normed` is set to `true`, then the coefficients are 
normalized accordingly.

Reference: Abramowitz and Stegun, Handbook of Mathematical Functions (Orthogonal Polynomials)

"""
function rc_legendre01(N::Int, normed=false)
    @assert N>=0 "parameter(s) out of range"
    α = [(2 * (n - 1) + 1) // (n) for n in 1:N]
    β = [(n - 1) // (n) for n in 1:N]
    return α, β
end


