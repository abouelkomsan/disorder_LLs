# ---------------------------------------------------------------------
# 5.  current operators and the matrix-element matrix
# ---------------------------------------------------------------------
"""
    apply_j!(out, U, nLL, Nk, dir)

`out .= J_dir * U` exploiting that J is block-bidiagonal in the LL index and
the identity in k.  Costs O(dim^2) instead of the O(dim^3) of a dense product.
`dir = :x` or `:y`.
"""
apply_j(U::AbstractMatrix{ComplexF64}, nLL::Int, Nk::Int, dir::Symbol) =
    apply_j!(zeros(ComplexF64, size(U)), U, nLL, Nk, dir)

function apply_j!(out::AbstractMatrix{ComplexF64}, U::AbstractMatrix{ComplexF64},
                  nLL::Int, Nk::Int, dir::Symbol)
    fill!(out, 0)
    @inbounds for mm in 1:nLL
        r = (mm-1)*Nk+1 : mm*Nk
        if mm+1 <= nLL
            c = dir === :x ? ComplexF64(sqrt(mm)) : ComplexF64(-im*sqrt(mm))
            @views out[r,:] .+= c .* U[mm*Nk+1:(mm+1)*Nk, :]
        end
        if mm-1 >= 1
            c = dir === :x ? ComplexF64(sqrt(mm-1)) : ComplexF64(im*sqrt(mm-1))
            @views out[r,:] .+= c .* U[(mm-2)*Nk+1:(mm-1)*Nk, :]
        end
    end
    return out
end

"""
    current_matrices(U, nLL, Nk) -> (Ax, Ay)

`Ax[a,b] = <a|jx|b>`, `Ay[a,b] = <a|jy|b>` (both Hermitian).
"""
function current_matrices(U::AbstractMatrix{ComplexF64}, nLL::Int, Nk::Int)
    Ax = U' * apply_j(U, nLL, Nk, :x)
    Ay = U' * apply_j(U, nLL, Nk, :y)
    return Ax, Ay
end

"""
    sigma_elements(Aa, Ab) -> M with M[a,b] = <a|ja|b><b|jb|a>

Same as `calc_sigma_elements2` but without recomputing `U'*J*U`.
`M` is Hermitian, so `Im M` is antisymmetric -- this is what the O(dim^2)
chemical-potential sweeps below rely on.
"""
sigma_elements(Aa::AbstractMatrix, Ab::AbstractMatrix) = Aa .* transpose(Ab)

"""
    sigma_elements!(M, Aa, Ab)

As `sigma_elements`, but exploiting that `Aa`, `Ab` are Hermitian, so
`transpose(Ab) == conj(Ab)` and the product is a plain elementwise one.
`Aa .* transpose(Ab)` reads `Ab` down its columns *and* across its rows: for
a 13 MB matrix that is a cache miss per element.  The conjugate form is a
single streaming pass.
"""
sigma_elements!(M::AbstractMatrix, Aa::AbstractMatrix, Ab::AbstractMatrix) =
    (M .= Aa .* conj.(Ab))

# ---------------------------------------------------------------------
# 6.  chemical-potential sweeps in O(dim^2) *total*
# ---------------------------------------------------------------------
#
#  All quantities have the form   F(mu) = sum_{a <= p < b} T[a,b]
#  with T antisymmetric (T[b,a] = -T[a,b]) and p = #(states below mu).
#  Then
#         F(p) = F(p-1) + sum_b T[p,b]
#  so a single row-sum pass (O(dim^2)) plus a cumulative sum gives F for
#  *every* mu at once, instead of O(dim^2) per mu value.
# ---------------------------------------------------------------------

"cumulative sum with Kahan compensation (the row sums can be large and cancel)"
function _kahan_cumsum(r::Vector{T}) where {T}
    out = Vector{T}(undef, length(r))
    s = zero(T); c = zero(T)
    @inbounds for i in eachindex(r)
        y = r[i] - c
        t = s + y
        c = (t - s) - y
        s = t
        out[i] = s
    end
    return out
end

"""
    hall_sumrule_curve(M, E; tol=1e-12, cutoff=Inf) -> F

`F[p]` = `Im sum_{a<=p<b} M[a,b]/(E[b]-E[a])^2`, i.e. exactly `sumrule2`
(and `partial_sumrule2` when `cutoff` is finite) evaluated at *every* possible
chemical potential at once.  `E` must be sorted ascending.

Use `hall_sumrule_at(F, E, mu)` to read it off at a given mu.
"""
function hall_sumrule_curve(M::AbstractMatrix, E::AbstractVector{<:Real};
                            tol::Real=1e-12, cutoff::Real=Inf)
    d = length(E)
    r = zeros(Float64, d)
    @inbounds for b in 1:d
        Eb = E[b]
        @simd for a in 1:d
            dE = Eb - E[a]
            w  = ifelse(abs(dE) < tol || abs(dE) > cutoff, 0.0,
                        imag(M[a,b])/(dE*dE))
            r[a] += w
        end
    end
    return _kahan_cumsum(r)
end

"""
    sigma_curve(omega, M, E, delta; tol=1e-12) -> Fs

`Fs[p]` = sigma(omega, mu) with `p` states occupied, for every p at once.
Identical definition to `sigma2` in disorder_LL.jl (including the overall `i`).
Complex: real and imaginary parts are both meaningful.
"""
function sigma_curve(omega::Real, M::AbstractMatrix, E::AbstractVector{<:Real},
                     delta::Real; tol::Real=1e-12, hermitian::Bool=true)
    d = length(E)
    r = zeros(ComplexF64, d)
    w = ComplexF64(omega, delta)
    # `M` is Hermitian for every conductivity built by `sigma_elements`, so
    # M[b,a] == conj(M[a,b]) and the column-strided read can be dropped --
    # with a 13 MB matrix that read cost a cache miss per term.
    if hermitian
        @inbounds for b in 1:d
            Eb = E[b]
            @simd for a in 1:d
                dE  = Eb - E[a]
                Mab = M[a,b]
                t   = (Mab/(w - dE) + conj(Mab)/(w + dE))/dE
                r[a] += ifelse(abs(dE) < tol, zero(t), t)
            end
        end
    else
        @inbounds for b in 1:d
            Eb = E[b]
            for a in 1:d
                dE = Eb - E[a]
                (abs(dE) < tol) && continue
                r[a] += (M[a,b]/(w - dE) + M[b,a]/(w + dE))/dE
            end
        end
    end
    return im .* _kahan_cumsum(r)
end

"""index p = number of eigenvalues <= mu"""
@inline occ_index(E::AbstractVector{<:Real}, mu::Real) = searchsortedlast(E, mu)

"""read a curve F (indexed by occupation p, p=1..d) at chemical potential mu"""
@inline function curve_at(F::AbstractVector, E::AbstractVector{<:Real}, mu::Real)
    p = searchsortedlast(E, mu)
    return p == 0 ? zero(eltype(F)) : F[p]
end

curve_at(F::AbstractVector, E::AbstractVector{<:Real}, mus::AbstractVector) =
    [curve_at(F, E, mu) for mu in mus]

# ---------------------------------------------------------------------
# 7.  full frequency spectra:  histogram in dE + Lorentzian convolution
# ---------------------------------------------------------------------
"""
    sigma_spectrum(omegas, mus, M, E, delta; refine=4, tol=1e-12)

sigma(omega, mu) on a *uniform* `omegas` grid for each mu in `mus`,
returned as a `length(omegas) x length(mus)` complex matrix.

Method: for fixed mu,

    sigma(w) = i * sum_p [ w+_p/(w - dE_p + i d) + w-_p/(w + dE_p + i d) ]

is a convolution of the spectral weight with the kernel `1/(x + i d)`.  The
weights are binned on a uniform energy grid (linear deposit, `refine` times
finer than the omega grid) and convolved with one FFT, turning the
`O(N_omega * dim^2)` double loop of `sigma2` into `O(dim^2 + N log N)`.

Set `refine` larger if `delta` is comparable to the omega spacing.
"""
function sigma_spectrum(omegas::AbstractRange, mus::AbstractVector,
                        M::AbstractMatrix, E::AbstractVector{<:Real},
                        delta::Real; refine::Int=4, tol::Real=1e-12)
    d  = length(E)
    dw = step(omegas)
    h  = dw/refine
    Emax = maximum(E) - minimum(E)
    nb   = ceil(Int, Emax/h) + 2
    ngrid = 2nb + 1                      # bins centred on  (-nb:nb)*h
    out = Matrix{ComplexF64}(undef, length(omegas), length(mus))

    # kernel on the same lattice, wide enough to cover omega - Egrid
    nk_lo = floor(Int, (first(omegas))/h) - nb
    nk_hi = ceil(Int,  (last(omegas))/h)  + nb
    K = ComplexF64[ 1/(k*h + im*delta) for k in nk_lo:nk_hi ]

    # ---- one FFT plan and one kernel transform for all mu -------------
    nconv = ngrid + length(K) - 1
    nfft  = HAVE_FFTW ? nextprod((2,3,5,7), nconv) : nconv
    if HAVE_FFTW
        pf  = plan_fft!(Vector{ComplexF64}(undef, nfft))
        pi_ = plan_ifft!(Vector{ComplexF64}(undef, nfft))
        Kf  = zeros(ComplexF64, nfft); Kf[1:length(K)] .= K; pf*Kf
        buf = Vector{ComplexF64}(undef, nfft)
    end
    # index into the convolution for each requested omega
    cidx = [ (round(Int, w/h) + nb - nk_lo) + 1 for w in omegas ]

    W = zeros(ComplexF64, ngrid)
    for (jmu, mu) in enumerate(mus)
        p = searchsortedlast(E, mu)
        fill!(W, 0)
        if p > 0 && p < d
            @inbounds for a in 1:p, b in (p+1):d
                dE = E[b] - E[a]
                abs(dE) < tol && continue
                x  = dE/h
                i0 = floor(Int, x); f = x - i0
                wp = M[a,b]/dE
                wm = conj(M[a,b])/dE        # M Hermitian; avoids the strided read
                W[nb+1+i0] += wp*(1-f)          # weight at +dE
                W[nb+2+i0] += wp*f
                W[nb+1-i0] += wm*(1-f)          # weight at -dE
                W[nb-i0]   += wm*f
            end
        end
        if HAVE_FFTW
            fill!(buf, 0); @views buf[1:ngrid] .= W
            pf*buf; buf .*= Kf; pi_*buf
            @inbounds for i in eachindex(cidx)
                out[i,jmu] = im*buf[cidx[i]]
            end
        else
            S = _convolve_direct(W, K, nk_lo, nb, omegas, h)
            @inbounds for i in eachindex(omegas)
                out[i,jmu] = im*S[i]
            end
        end
    end
    return out
end

function _convolve_direct(W, K, nk_lo, nb, omegas, h)
    nW = length(W); nK = length(K)
    S  = Vector{ComplexF64}(undef, length(omegas))
    @inbounds for (i,w) in enumerate(omegas)
        iw = round(Int, w/h); s = zero(ComplexF64)
        for j in 1:nW
            kidx = iw - (j-1-nb) - nk_lo + 1
            (1 <= kidx <= nK) && (s += W[j]*K[kidx])
        end
        S[i] = s
    end
    return S
end

"""density of states (Lorentzian broadened), all omegas at once"""
function dos_curve(omegas::AbstractVector, E::AbstractVector{<:Real}, delta::Real)
    out = Vector{Float64}(undef, length(omegas))
    @inbounds for (i,w) in enumerate(omegas)
        s = 0.0
        @simd for e in E
            s += delta/((w-e)^2 + delta^2)
        end
        out[i] = s
    end
    return out          # = -Im sum 1/(w-E+i delta)
end
