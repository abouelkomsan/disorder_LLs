# One disorder configuration, and the quantised Hall plateaux.
#   julia --project=. examples/quickstart.jl
using DisorderLL, LinearAlgebra, Random, Printf

m   = LLModel(5, lattice(15,0,0,15))          # 5 Landau levels, 15x15 supercell
imp = gen_imp_pm(m, 2000; rng=Xoshiro(1))     # 2000 (+V0,-V0) impurity pairs
@printf("N_phi = %d, dim = %d\n", m.Nk, m.dim)

E, Mxx, Mxy = solve_config(m, imp, 0.1; xi=0.5)
Er = real.(E)

# every chemical potential at once, in one O(dim^2) pass
F = hall_sumrule_curve(Mxy, Er)
println("\n  mu    sigma_xy [e^2/h]   Re sigma_xx(w=0.05) [e^2/h]")
for mu in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    sxx = real(curve_at(sigma_curve(0.05, Mxx, Er, 0.005), Er, mu))/(2*m.Nk)
    @printf("%5.1f %14.4f %22.4f\n", mu, curve_at(F, Er, mu)/m.Nk, sxx)
end
println("\n(the gaps at mu = 1, 2, 3 should sit close to 1, 2, 3;")
println(" a single configuration is noisy -- see disorder_average.jl)")
