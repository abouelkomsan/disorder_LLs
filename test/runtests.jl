using DisorderLL
using Test, LinearAlgebra, Random, Statistics

# =====================================================================
#  Reference implementations.
#
#  These are the ORIGINAL, unoptimised formulas (naive double loops over
#  occupied/empty pairs, one chemical potential at a time).  The package
#  replaces them with an O(dim^2)-total cumulative-sum scheme and an
#  FFT-convolved spectral method; these tests check the replacements
#  reproduce the originals.
# =====================================================================
function ref_sumrule(mu, M, E)
    s = 0.0 + 0.0im
    p = searchsortedlast(E, mu)
    p == 0 && return 0.0
    for a in 1:p, b in (p+1):length(E)
        s += M[a,b] / (E[b]-E[a])^2
    end
    return imag(s)
end
function ref_sigma(omega, mu, M, E, delta)
    s = 0.0 + 0.0im
    p = searchsortedlast(E, mu)
    p == 0 && return 0.0 + 0.0im
    for a in 1:p, b in (p+1):length(E)
        dE = E[b]-E[a]
        s += (M[a,b]/(omega - dE + im*delta) + M[b,a]/(omega + dE + im*delta))/dE
    end
    return s*im
end

@testset "DisorderLL" begin

# ---------------------------------------------------------------------
@testset "generalised Laguerre" begin
    # L_0 = 1 ; L_1^a = 1+a-x ; L_2^a = x^2/2 - (a+2)x + (a+1)(a+2)/2
    for a in 0:4, x in (0.0, 0.7, 3.1, 11.0)
        @test laguerre_gen(0, a, x) ≈ 1.0
        @test laguerre_gen(1, a, x) ≈ 1 + a - x
        @test laguerre_gen(2, a, x) ≈ x^2/2 - (a+2)*x + (a+1)*(a+2)/2
    end
end

# ---------------------------------------------------------------------
@testset "clean limit" begin
    m = LLModel(3, lattice(4,0,0,4))
    @test m.Nk == 16 && m.dim == 48
    imp = ImpConfig(zeros(2,1), [0.0])              # no disorder
    H = build_H(m, imp, 0.0)
    E = eigvals(H)
    # exactly nLL levels at n+1/2, each N_phi-fold degenerate
    for n in 0:2
        @test count(e -> abs(e - (n+0.5)) < 1e-10, E) == m.Nk
    end
end

# ---------------------------------------------------------------------
@testset "Hamiltonian is Hermitian" begin
    m   = LLModel(3, lattice(4,0,0,4))
    imp = gen_imp_pm(m, 20; rng=Xoshiro(1))
    H   = Matrix(build_H(m, imp, 0.2; xi=0.3))
    @test maximum(abs, H - H') < 1e-12
end

# ---------------------------------------------------------------------
@testset "current matrices and sigma elements" begin
    m   = LLModel(4, lattice(4,0,0,4))
    imp = gen_imp_pm(m, 15; rng=Xoshiro(7))
    F   = eigen(build_H(m, imp, 0.15; xi=0.4))
    U   = Matrix(F.vectors)
    Ax, Ay = current_matrices(U, m.nLL, m.Nk)
    @test maximum(abs, Ax - Ax') < 1e-10          # currents are Hermitian
    @test maximum(abs, Ay - Ay') < 1e-10
    # sigma_elements uses transpose; the fast path uses conj (valid because
    # Ax, Ay are Hermitian).  They must agree.
    Mxy = sigma_elements(Ax, Ay)
    Mxy2 = similar(Mxy); sigma_elements!(Mxy2, Ax, Ay)
    @test maximum(abs, Mxy - Mxy2) < 1e-12
    @test maximum(abs, Mxy - Mxy') < 1e-10        # M is Hermitian
end

# ---------------------------------------------------------------------
@testset "chemical-potential sweeps vs naive formulas" begin
    m   = LLModel(4, lattice(4,0,0,4))
    imp = gen_imp_pm(m, 15; rng=Xoshiro(11))
    E, Mxx, Mxy = solve_config(m, imp, 0.15; xi=0.4)
    Er  = real.(E)
    mus = range(Er[1]-0.05, Er[end]+0.05; length=41)

    Fsr = hall_sumrule_curve(Mxy, Er)
    for mu in mus
        @test curve_at(Fsr, Er, mu) ≈ ref_sumrule(mu, Mxy, Er) atol=1e-9
    end
    for w in (0.0, 0.37, 1.4), delta in (0.02,)
        cxy = sigma_curve(w, Mxy, Er, delta)
        cxx = sigma_curve(w, Mxx, Er, delta)
        for mu in mus
            @test curve_at(cxy, Er, mu) ≈ ref_sigma(w, mu, Mxy, Er, delta) atol=1e-8
            @test curve_at(cxx, Er, mu) ≈ ref_sigma(w, mu, Mxx, Er, delta) atol=1e-8
        end
    end
end

# ---------------------------------------------------------------------
@testset "spectra (binned + FFT) vs exact" begin
    m   = LLModel(4, lattice(4,0,0,4))
    imp = gen_imp_pm(m, 15; rng=Xoshiro(13))
    E, Mxx, Mxy = solve_config(m, imp, 0.15; xi=0.4)
    Er = real.(E)
    ws = 0.0:0.004:2.0
    mus = [Er[div(end,3)]+1e-9, Er[div(2*end,3)]+1e-9]
    S  = sigma_spectrum(ws, mus, Mxy, Er, 0.03; refine=8)
    scale = maximum(abs, S)
    for (j,mu) in enumerate(mus), i in 1:20:length(ws)
        @test S[i,j] ≈ ref_sigma(ws[i], mu, Mxy, Er, 0.03) atol=5e-3*scale
    end
end

# ---------------------------------------------------------------------
@testset "workspace path is equivalent and reusable" begin
    m   = LLModel(3, lattice(4,0,0,4))
    imp = gen_imp_pm(m, 12; rng=Xoshiro(17))
    E1, Mxx1, Mxy1 = solve_config(m, imp, 0.2; xi=0.5)
    ws = LLWorkspace(m, size(imp.pos,2))
    E2, Mxx2, Mxy2 = solve_config!(ws, m, imp, 0.2; xi=0.5)
    @test E1 ≈ E2
    @test maximum(abs, Mxy1 - Mxy2) < 1e-12
    E3, Mxx3, Mxy3 = solve_config!(ws, m, imp, 0.2; xi=0.5)   # reuse
    @test E1 ≈ E3
    @test maximum(abs, Mxy1 - Mxy3) < 1e-12
end

# ---------------------------------------------------------------------
@testset "physics: quantised Hall plateaux and vanishing Im sigma_xy(0)" begin
    # nLL levels sit at 1/2, 3/2, ... so only the gaps BELOW the topmost
    # level are physical: with nLL levels retained there are nLL-1 plateaux.
    # (mu above every level fills the truncated space and gives sigma_xy = 0,
    # correctly, since there are then no empty states to make transitions to.)
    m = LLModel(4, lattice(8,0,0,8))
    Nphi = m.Nk
    ws = LLWorkspace(m, 2*300)
    n = 12
    acc = zeros(Float64, 3)
    for ic in 1:n
        imp = gen_imp_pm(m, 300; rng=Xoshiro(1000+ic))
        E, Mxx, Mxy = solve_config!(ws, m, imp, 0.1; xi=0.5)
        Er = real.(E)
        c = hall_sumrule_curve(Mxy, Er)
        for (k,mu) in enumerate((1.0, 2.0, 3.0))
            acc[k] += curve_at(c, Er, mu)/Nphi
        end
        # Im sigma_xy(omega=0) vanishes identically
        s0 = sigma_curve(0.0, Mxy, Er, 0.01)
        @test maximum(abs, imag.(s0)) < 1e-9 * max(1.0, maximum(abs, real.(s0)))
    end
    acc ./= n
    for (k, nu) in enumerate((1,2,3))
        @test acc[k] ≈ nu rtol=0.05      # plateau at nu e^2/h
    end
end

# ---------------------------------------------------------------------
@testset "density of states normalisation" begin
    m = LLModel(3, lattice(6,0,0,6))
    imp = gen_imp_pm(m, 100; rng=Xoshiro(23))
    E, _, _ = solve_config(m, imp, 0.1; xi=0.5)
    eg = collect(-1.0:0.002:6.0)
    d  = dos_curve(eg, real.(E), 0.01)
    # integral DOS dE / pi = number of states
    @test sum(d)*(eg[2]-eg[1])/pi ≈ m.dim rtol=0.01
end

end
