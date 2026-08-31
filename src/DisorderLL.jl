"""
    DisorderLL

Disorder-averaged transport in Landau levels: exact diagonalisation of

    H = ħω_c (n + 1/2) + V(r)

projected onto `nLL` Landau levels on a magnetic-Brillouin-zone torus, with
randomly placed impurities, and the Kubo conductivities σ_xx(ω,μ), σ_xy(ω,μ)
and the Hall (TKNN) sum rule averaged over disorder realisations.

Quick start:

```julia
using DisorderLL

m   = LLModel(5, lattice(20,0,0,20))        # 5 Landau levels, 20x20 supercell
imp = gen_imp_pm(m, 3556)                    # 3556 +/- impurity pairs
E, Mxx, Mxy = solve_config(m, imp, 0.1; xi=0.5)

F   = hall_sumrule_curve(Mxy, real.(E))      # sigma_xy for EVERY mu at once
curve_at(F, real.(E), 1.0) / m.Nk            # -> ~1, the nu=1 plateau
```

or, for a full disorder average, `run_disorder(...)`.

See the README for conventions, units and the convergence requirements
(broadening, sample count, system size).
"""
module DisorderLL

using LinearAlgebra
using Random
using Distributions
using JLD2
using Printf
using Statistics
using Dates

const HAVE_FFTW = try
    @eval using FFTW
    true
catch
    false
end

# --- lattice / magnetic Brillouin zone --------------------------------
export lattice, kpoints, g1g2, projector
# --- model, disorder, Hamiltonian -------------------------------------
export LLModel, LLWorkspace, ImpConfig, supercell
export gen_imp_pm, gen_imp_rand, Vtable, Vtable!, build_H, build_H!
export laguerre_gen
# --- observables -------------------------------------------------------
export apply_j, apply_j!, current_matrices, sigma_elements, sigma_elements!
export hall_sumrule_curve, sigma_curve, sigma_spectrum, dos_curve
export curve_at, occ_index
export solve_config, solve_config!
# --- drivers -----------------------------------------------------------
export run_disorder, run_spectra, merge_avg, SAVDIR

include("lattice.jl")
include("helpers.jl")
include("model.jl")
include("disorder.jl")
include("hamiltonian.jl")
include("conductivity.jl")
include("solve.jl")
include("runs.jl")

end # module
