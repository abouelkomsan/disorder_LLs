# ---------------------------------------------------------------------
# 4.  the Hamiltonian
# ---------------------------------------------------------------------
"""
    build_H(m, V; threaded=false)

Assemble the `dim x dim` Hamiltonian from the tabulated `V(q)`.
Inner loop is 2 complex multiplies + 1 add per (k1,k2,b,LL-pair); all
transcendentals and all Laguerre evaluations have been hoisted into `LLModel`.
"""
function build_H(m::LLModel, V::Vector{ComplexF64}; threaded::Bool=false)
    npair = length(m.pairs)
    return build_H!(zeros(ComplexF64, m.dim, m.dim),
                    Matrix{ComplexF64}(undef, npair, m.nq),
                    m, V; threaded=threaded)
end

"""in-place `build_H`; `H` (dim x dim) and `VF` (npair x nq) are caller-owned"""
function build_H!(H::Matrix{ComplexF64}, VF::Matrix{ComplexF64},
                  m::LLModel, V::Vector{ComplexF64}; threaded::Bool=false)
    npair = length(m.pairs); nb = length(m.blabel); Nk = m.Nk
    @inbounds for idx in 1:m.nq
        v = V[idx]
        @simd for ip in 1:npair
            VF[ip,idx] = m.Ftab[ip,idx]*v
        end
    end

    fill!(H, 0)
    body = function (kind2)
        acc = Vector{ComplexF64}(undef, npair)
        k2  = m.klabel[kind2]
        @inbounds for kind1 in 1:Nk
            k1 = m.klabel[kind1]
            iu = qindex(m, k1[1]-k2[1], k1[2]-k2[2])
            fill!(acc, 0)
            for ib in 1:nb
                c   = m.EPb[ib,kind1]*m.Pbk[ib,kind2]
                idx = iu + m.boff[ib]
                @simd for ip in 1:npair
                    acc[ip] += VF[ip,idx]*c
                end
            end
            f = m.Pk[kind1,kind2]*m.scale
            for ip in 1:npair
                n1, n2 = m.pairs[ip]
                val = acc[ip]*f
                H[Nk*(n1-1)+kind1, Nk*(n2-1)+kind2] = val
                if n1 != n2
                    H[Nk*(n2-1)+kind2, Nk*(n1-1)+kind1] = conj(val)
                end
            end
        end
    end
    if threaded
        Threads.@threads for kind2 in 1:Nk; body(kind2); end
    else
        for kind2 in 1:Nk; body(kind2); end
    end

    @inbounds for n in 1:m.nLL, k in 1:Nk
        H[Nk*(n-1)+k, Nk*(n-1)+k] += (n - 1 + 0.5)
    end
    return Hermitian(H)
end

"convenience: model + impurity configuration -> H"
build_H(m::LLModel, imp::ImpConfig, V0::Real; xi::Real=0.0, threaded::Bool=false) =
    build_H(m, Vtable(m, imp, V0; xi=xi); threaded=threaded)
