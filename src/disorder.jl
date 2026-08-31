# ---------------------------------------------------------------------
# 2.  disorder realisations
# ---------------------------------------------------------------------
"""
    ImpConfig(pos, str)

`pos` is a `2 x nimp` matrix of impurity positions in the real-space supercell,
`str` the (fixed!) impurity strengths.

NOTE on a bug in the original code: `V_rand(q,V0,imp_loc)` in disorder_LL.jl
draws `impsgn = rand([-1,1])` *inside* the q-loop, so the sign of every
impurity is re-randomised for every matrix element.  The resulting matrix is
not the Fourier transform of any potential.  Here the strengths are drawn once
per configuration, which is both correct and what makes the fast
V(q)-tabulation possible.
"""
struct ImpConfig
    pos :: Matrix{Float64}      # 2 x nimp
    str :: Vector{Float64}      # nimp
end

"real-space supercell vectors (a1,a2 dual to G1,G2: a_i . G_j = 2pi delta_ij)"
function supercell(m::LLModel; a1=m.a0*[1.0,0.0], a2=m.a0*[0.0,1.0])
    T1 = m.klattice[1,1]*a1 + m.klattice[1,2]*a2
    T2 = m.klattice[2,1]*a1 + m.klattice[2,2]*a2
    return T1, T2
end

"""
    gen_imp_pm(m, nimp; rng)

`nimp` positive plus `nimp` negative impurities (the `imp_loc_plus /
imp_loc_minus` convention of disorder_LL.jl), strengths +-V0 applied later.
"""
function gen_imp_pm(m::LLModel, nimp::Int; rng=Random.default_rng(), kwargs...)
    T1, T2 = supercell(m; kwargs...)
    pos = Matrix{Float64}(undef, 2, 2nimp)
    @inbounds for j in 1:2nimp
        u = rand(rng); v = rand(rng)
        pos[1,j] = u*T1[1] + v*T2[1]
        pos[2,j] = u*T1[2] + v*T2[2]
    end
    str = [ j <= nimp ? 1.0 : -1.0 for j in 1:2nimp ]
    return ImpConfig(pos, str)
end

"""`nimp` impurities with random sign +-1 (fixed per configuration)."""
function gen_imp_rand(m::LLModel, nimp::Int; rng=Random.default_rng(), kwargs...)
    T1, T2 = supercell(m; kwargs...)
    pos = Matrix{Float64}(undef, 2, nimp)
    @inbounds for j in 1:nimp
        u = rand(rng); v = rand(rng)
        pos[1,j] = u*T1[1] + v*T2[1]
        pos[2,j] = u*T1[2] + v*T2[2]
    end
    str = rand(rng, (-1.0, 1.0), nimp)
    return ImpConfig(pos, str)
end

# ---------------------------------------------------------------------
# 3.  V(q) on the whole q-grid, in O(nq * nimp) BLAS flops
# ---------------------------------------------------------------------
"""
    Vtable(m, imp, V0; xi=0.0)

Returns `V[iq] = V0 * f(|q|) * sum_j s_j exp(-i q . r_j)` for every point of
the integer q grid, where `f = 1` (delta impurities) or
`f = exp(-|q|^2 xi^2/2)` (gaussian impurities).

The separability `exp(-i q.r) = A_r^{q1} B_r^{q2}` with `A_r = exp(-i g1.r)`,
`B_r = exp(-i g2.r)` turns the whole table into one `zgemm`.  The original
code evaluated the impurity sum *inside* the matrix-element loop, i.e.
`dim^2/2 * (2M+1)^2 * nimp` complex exponentials per configuration.
"""
function Vtable(m::LLModel, imp::ImpConfig, V0::Real; xi::Real=0.0)
    nimp = size(imp.pos, 2)
    nq2  = m.nq ÷ m.nq1
    return Vtable!(Vector{ComplexF64}(undef, m.nq),
                   Matrix{ComplexF64}(undef, m.nq1, nimp),
                   Matrix{ComplexF64}(undef, nq2,   nimp),
                   m, imp, V0; xi=xi)
end

"""in-place `Vtable`; `V`, `P`, `Qt` are caller-owned scratch"""
function Vtable!(V::Vector{ComplexF64}, P::Matrix{ComplexF64}, Qt::Matrix{ComplexF64},
                 m::LLModel, imp::ImpConfig, V0::Real; xi::Real=0.0)
    nimp = size(imp.pos, 2)
    nq1  = m.nq1; nq2 = m.nq ÷ m.nq1
    q1r  = m.q1range; q2r = m.q2range
    @assert size(P) == (nq1, nimp) && size(Qt) == (nq2, nimp) && length(V) == m.nq "Vtable! workspace was built for a different nimp / model"

    gr1 = Vector{Float64}(undef, nimp)   # g1 . r
    gr2 = Vector{Float64}(undef, nimp)   # g2 . r
    @inbounds for j in 1:nimp
        gr1[j] = m.g1[1]*imp.pos[1,j] + m.g1[2]*imp.pos[2,j]
        gr2[j] = m.g2[1]*imp.pos[1,j] + m.g2[2]*imp.pos[2,j]
    end

    # Both tables are stored with the impurity index *last* so that the
    # inner loop writes contiguously (writing Q[j,i] with j the impurity
    # index strides by nimp and costs a cache miss per element -- that
    # alone dominated the whole routine).
    @inbounds for j in 1:nimp
        z = cis(-gr1[j]); w = z^first(q1r); s = imp.str[j]
        for i in 1:nq1
            P[i,j] = s*w; w *= z
        end
        z = cis(-gr2[j]); w = z^first(q2r)
        for i in 1:nq2
            Qt[i,j] = w; w *= z
        end
    end

    mul!(reshape(V, nq1, nq2), P, transpose(Qt))
    if xi != 0
        @inbounds for idx in 1:m.nq
            V[idx] *= exp(-m.qnorm2[idx]*xi^2/2)
        end
    end
    V .*= V0
    return V
end
