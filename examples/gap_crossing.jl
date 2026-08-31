# Im sigma_xy(omega) as mu crosses from one Landau level to the next, and
# the normalisation check: the Hall sum rule and the omega->0 Kubo result
# must agree (up to the documented factor of 2) on the plateaux.
using DisorderLL, Random, Printf, Statistics

m  = LLModel(5, lattice(15,0,0,15))
ws = LLWorkspace(m, 2*2000)
Nphi = m.Nk
N  = parse(Int, get(ENV, "NCONFIG", "100"))
mus = [0.62, 0.90, 1.35]
wg  = 0.0:0.004:2.0
acc = zeros(ComplexF64, length(wg), length(mus))
sr  = zeros(Float64, 3); kb = zeros(Float64, 3)

for ic in 1:N
    imp = gen_imp_pm(m, 2000; rng=Xoshiro(ic))
    E, Mxx, Mxy = solve_config!(ws, m, imp, 0.1; xi=0.5)
    Er = real.(E)
    acc .+= sigma_spectrum(wg, mus, Mxy, Er, 0.001; refine=8)
    F = hall_sumrule_curve(Mxy, Er)
    c0 = sigma_curve(0.0, Mxy, Er, 0.001)
    for (k,mu) in enumerate((1.0, 2.0, 3.0))
        sr[k] += curve_at(F, Er, mu)/Nphi/N
        kb[k] += real(curve_at(c0, Er, mu))/(2Nphi)/N
    end
end
acc ./= (N * 2 * Nphi)     # -> e^2/h  (see README: sigma_curve/sigma_spectrum carry a factor 2)

println("normalisation check on the plateaux (should be 1, 2, 3):")
println("   nu    sum rule / N_phi     Re sigma_xy(0) / 2N_phi")
for k in 1:3
    @printf("   %d  %16.5f %24.5f\n", k, sr[k], kb[k])
end

println("\nlow-frequency Im sigma_xy, averaged over 0 < omega <= 0.05  [e^2/h]:")
lm = (wg .> 0) .& (wg .<= 0.05)
for (k,mu) in enumerate(mus)
    v = mean(imag.(acc[lm,k]))
    @printf("   mu=%.2f : %+.5f  (%s)\n", mu, v, v > 0 ? "positive" : "negative")
end
println("\nThe sign flips across the gap: positive above the n=0 peak,")
println("~zero in the gap, negative below the n=1 peak.")
