# ---------------------------------------------------------------------
# 8.  one full disorder configuration
# ---------------------------------------------------------------------
"""
    solve_config(m, imp, V0; xi, threaded) -> (E, Mxx, Mxy)

Build H, diagonalise, and return the eigenvalues together with the
conductivity matrix elements.  Eigenvectors are *not* returned: everything the
conductivity needs is contained in `Mxx` and `Mxy`.
Allocating version of `solve_config!`; prefer the latter in hot loops.
"""
function solve_config(m::LLModel, imp::ImpConfig, V0::Real;
                      xi::Real=0.0, threaded::Bool=false)
    ws = LLWorkspace(m, size(imp.pos, 2))
    return solve_config!(ws, m, imp, V0; xi=xi, threaded=threaded)
end

"""
    LLWorkspace(m, nimp)

Per-worker scratch space for `solve_config!`.  A configuration needs several
`dim x dim` complex arrays (H, J*U, Ax, Ay, M_xx, M_xy) plus the V tables;
reusing them leaves the LAPACK eigenvector matrix as the only large
allocation per configuration, which matters when many threads run at once.
"""
struct LLWorkspace
    H  :: Matrix{ComplexF64}
    VF :: Matrix{ComplexF64}
    V  :: Vector{ComplexF64}
    P  :: Matrix{ComplexF64}
    Qt :: Matrix{ComplexF64}
    JU :: Matrix{ComplexF64}
    Ax :: Matrix{ComplexF64}
    Ay :: Matrix{ComplexF64}
    Mxx:: Matrix{ComplexF64}
    Mxy:: Matrix{ComplexF64}
end

function LLWorkspace(m::LLModel, nimp::Int)
    d = m.dim; nq2 = m.nq ÷ m.nq1
    LLWorkspace(zeros(ComplexF64, d, d),
                Matrix{ComplexF64}(undef, length(m.pairs), m.nq),
                Vector{ComplexF64}(undef, m.nq),
                Matrix{ComplexF64}(undef, m.nq1, nimp),
                Matrix{ComplexF64}(undef, nq2,   nimp),
                Matrix{ComplexF64}(undef, d, d),
                Matrix{ComplexF64}(undef, d, d),
                Matrix{ComplexF64}(undef, d, d),
                Matrix{ComplexF64}(undef, d, d),
                Matrix{ComplexF64}(undef, d, d))
end

"""
    solve_config!(ws, m, imp, V0; xi, threaded) -> (E, Mxx, Mxy)

Build H, diagonalise, and form the conductivity matrix elements
`Mxx[a,b] = |<a|jx|b>|^2` and `Mxy[a,b] = <a|jx|b><b|jy|a>` reusing `ws`.
The returned matrices alias `ws.Mxx` / `ws.Mxy` -- consume them before the
next call on the same workspace.
"""
function solve_config!(ws::LLWorkspace, m::LLModel, imp::ImpConfig, V0::Real;
                       xi::Real=0.0, threaded::Bool=false)
    Vtable!(ws.V, ws.P, ws.Qt, m, imp, V0; xi=xi)
    build_H!(ws.H, ws.VF, m, ws.V; threaded=threaded)
    F = eigen!(Hermitian(ws.H))
    E = F.values
    U = F.vectors
    apply_j!(ws.JU, U, m.nLL, m.Nk, :x); mul!(ws.Ax, U', ws.JU)
    apply_j!(ws.JU, U, m.nLL, m.Nk, :y); mul!(ws.Ay, U', ws.JU)
    sigma_elements!(ws.Mxx, ws.Ax, ws.Ax)
    sigma_elements!(ws.Mxy, ws.Ax, ws.Ay)
    return E, ws.Mxx, ws.Mxy
end
