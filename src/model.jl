# ---------------------------------------------------------------------
# 1.  the model: everything that does NOT depend on the disorder sample
# ---------------------------------------------------------------------
"""
    LLModel(nLL, klattice; M, a0, G1, G2)

Pre-computes every disorder-independent ingredient of the Hamiltonian:

* `klabel`      integer momentum labels of the magnetic BZ
* `Ftab`        the LL form factor  exp(-|q|^2/4) * <n1|e^{iqr}|n2>  on the
                full integer q-grid, for every LL pair (n1<=n2)
* `Pk`, `EPb`, `Pbk`   the pure phase factors, as lookup tables

Building `LLModel` once and reusing it for every disorder configuration is
where a large part of the speed-up comes from: in the original code the
Laguerre polynomials, `projector` calls and `exp`s were recomputed inside the
innermost loop, i.e. `dim^2 * (2M+1)^2` times per configuration.
"""
struct LLModel
    nLL     :: Int
    klattice:: Matrix{Int}
    klabel  :: Vector{NTuple{2,Int}}
    Nk      :: Int
    dim     :: Int
    M       :: Int
    a0      :: Float64
    g1      :: Vector{Float64}
    g2      :: Vector{Float64}
    cg      :: Float64                     # cross(g1,g2)
    scale   :: Float64                     # 1/(sqrt(3)*a0*Nk/2)  (as in disorder_LL.jl)
    # --- q-grid bookkeeping -------------------------------------------------
    q1range :: UnitRange{Int}
    q2range :: UnitRange{Int}
    nq1     :: Int
    nq      :: Int
    qnorm2  :: Vector{Float64}             # |q|^2 on the grid (linear index)
    # --- b (reciprocal supercell) list --------------------------------------
    blabel  :: Vector{NTuple{2,Int}}
    boff    :: Vector{Int}                 # linear-index offset of b on q-grid
    eta     :: Vector{Float64}             # +-1 from projector(b/2)
    # --- tables -------------------------------------------------------------
    pairs   :: Vector{NTuple{2,Int}}       # (n1,n2) with n1<=n2
    Ftab    :: Matrix{ComplexF64}          # [npair, nq]
    Pk      :: Matrix{ComplexF64}          # [Nk,Nk]  cis(cg/2 * k1 x k2)
    EPb     :: Matrix{ComplexF64}          # [nb,Nk]  eta_b * cis(cg/2 * b x k)
    Pbk     :: Matrix{ComplexF64}          # [nb,Nk]  cis(cg/2 * b x k)
end

@inline qindex(m::LLModel, q1::Int, q2::Int) =
    (q2 - first(m.q2range))*m.nq1 + (q1 - first(m.q1range)) + 1

function LLModel(nLL::Int, klattice::AbstractMatrix; M::Int=2,
                 a0::Float64=sqrt(2pi),
                 G1::Vector{Float64}=(2pi/sqrt(2pi))*[1.0,0.0],
                 G2::Vector{Float64}=(2pi/sqrt(2pi))*[0.0,1.0])

    kl   = Int.(klattice)
    klab = [ (Int(k[1]), Int(k[2])) for k in kpoints(kl) ]
    Nk   = length(klab)
    dim  = nLL*Nk
    g1, g2, = g1g2(kl, G1, G2)
    g1 = Vector{Float64}(g1);  g2 = Vector{Float64}(g2)
    cg = xprod(g1, g2)

    # ---- b list, eta, and the q box ---------------------------------------
    blabel = NTuple{2,Int}[]
    eta    = Float64[]
    for m in -M:M, n in -M:M
        b = (m*kl[1,1] + n*kl[2,1], m*kl[1,2] + n*kl[2,2])
        push!(blabel, b)
        # exactly the criterion of disorder_LL.jl: eta = +1 iff b/2 folds to 0
        push!(eta, projector([b[1]/2, b[2]/2], kl)[1] == [0,0] ? 1.0 : -1.0)
    end
    nb = length(blabel)

    k1s = [k[1] for k in klab];  k2s = [k[2] for k in klab]
    b1s = [b[1] for b in blabel]; b2s = [b[2] for b in blabel]
    q1range = (minimum(k1s)-maximum(k1s)+minimum(b1s)) : (maximum(k1s)-minimum(k1s)+maximum(b1s))
    q2range = (minimum(k2s)-maximum(k2s)+minimum(b2s)) : (maximum(k2s)-minimum(k2s)+maximum(b2s))
    nq1 = length(q1range); nq2 = length(q2range); nq = nq1*nq2

    qvecx  = Vector{Float64}(undef, nq)
    qvecy  = Vector{Float64}(undef, nq)
    qnorm2 = Vector{Float64}(undef, nq)
    @inbounds for (j2, q2) in enumerate(q2range), (j1, q1) in enumerate(q1range)
        idx = (j2-1)*nq1 + j1
        x = q1*g1[1] + q2*g2[1]
        y = q1*g1[2] + q2*g2[2]
        qvecx[idx] = x; qvecy[idx] = y; qnorm2[idx] = x*x + y*y
    end

    # offset (not absolute index) of b on the linearised q grid, so that
    #   qindex(u+b) == qindex(u) + boff[b]
    boff = [ b[2]*nq1 + b[1] for b in blabel ]

    # ---- LL form factors on the whole q grid -------------------------------
    pairs = NTuple{2,Int}[]
    for n1 in 1:nLL, n2 in n1:nLL
        push!(pairs, (n1,n2))
    end
    npair = length(pairs)
    Ftab = Matrix{ComplexF64}(undef, npair, nq)
    @inbounds for (ip,(n1,n2)) in enumerate(pairs)
        nmin  = min(n1-1, n2-1)
        nmax  = max(n1-1, n2-1)
        ndiff = abs(n1-n2)
        pref  = sqrt_factratio(nmin, nmax)
        for idx in 1:nq
            x  = qnorm2[idx]
            qx = qvecx[idx]; qy = qvecy[idx]
            # branch of disorder_LL.jl: (n2>n1) -> ((qx-i qy)/sqrt2)^ndiff
            #                           (n2<=n1)-> (-(qx+i qy)/sqrt2)^ndiff
            z = n2 > n1 ? (qx - im*qy)/sqrt(2) : -(qx + im*qy)/sqrt(2)
            Ftab[ip, idx] = exp(-x/4) * pref * laguerre_gen(nmin, ndiff, x/2) * z^ndiff
        end
    end

    # ---- phase tables ------------------------------------------------------
    h = cg/2
    Pk = Matrix{ComplexF64}(undef, Nk, Nk)
    @inbounds for j in 1:Nk, i in 1:Nk
        Pk[i,j] = cis(h*xprod(klab[i], klab[j]))
    end
    Pbk = Matrix{ComplexF64}(undef, nb, Nk)
    EPb = Matrix{ComplexF64}(undef, nb, Nk)
    @inbounds for k in 1:Nk, ib in 1:nb
        p = cis(h*xprod(blabel[ib], klab[k]))
        Pbk[ib,k] = p
        EPb[ib,k] = eta[ib]*p
    end

    scale = 1 / (sqrt(3)*a0*Nk/2)

    return LLModel(nLL, kl, klab, Nk, dim, M, a0, g1, g2, cg, scale,
                   q1range, q2range, nq1, nq, qnorm2,
                   blabel, boff, eta, pairs, Ftab, Pk, EPb, Pbk)
end
