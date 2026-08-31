# ---------------------------------------------------------------------
# 0.  small helpers
# ---------------------------------------------------------------------
xprod(u::NTuple{2,<:Real}, v::NTuple{2,<:Real}) = u[1]*v[2] - u[2]*v[1]
xprod(u::AbstractVector, v::AbstractVector)     = u[1]*v[2] - u[2]*v[1]

"""
    laguerre_gen(n, alpha, x)

Generalised Laguerre polynomial L_n^{(alpha)}(x) by the standard three-term
recurrence.  Replaces `Laguerre{alpha}(e_n)(x)` from SpecialPolynomials, which
allocates a polynomial object on every call.
"""
@inline function laguerre_gen(n::Int, alpha::Int, x::Float64)
    n == 0 && return 1.0
    Lkm1 = 1.0
    Lk   = 1.0 + alpha - x
    n == 1 && return Lk
    @inbounds for k in 2:n
        Lkp1 = ((2k - 1 + alpha - x)*Lk - (k - 1 + alpha)*Lkm1) / k
        Lkm1 = Lk
        Lk   = Lkp1
    end
    return Lk
end

"""ratio sqrt(nmin! / nmax!) computed without overflow."""
@inline function sqrt_factratio(nmin::Int, nmax::Int)
    r = 1.0
    @inbounds for k in (nmin+1):nmax
        r /= k
    end
    return sqrt(r)
end
