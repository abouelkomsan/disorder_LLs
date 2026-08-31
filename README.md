# DisorderLL

Disorder-averaged transport in Landau levels, by exact diagonalisation.

Solves

$$H = \hbar\omega_c\left(n+\tfrac12\right) + V(\mathbf r)$$

projected onto `nLL` Landau levels on a magnetic-Brillouin-zone torus with
randomly placed impurities, and computes the Kubo conductivities
$\sigma_{xx}(\omega,\mu)$ and $\sigma_{xy}(\omega,\mu)$, the Hall (TKNN) sum
rule and the density of states, averaged over disorder realisations.

## Install

```julia
using Pkg
Pkg.develop(path="/path/to/disorder_LLs")
```

Julia ≥ 1.9. Dependencies: `Distributions`, `JLD2`, `FFTW` (optional — the
spectral routine falls back to direct convolution without it).

## Quick start

```julia
using DisorderLL

m   = LLModel(5, lattice(20,0,0,20))       # 5 Landau levels, 20x20 supercell
imp = gen_imp_pm(m, 3556)                   # 3556 (+V0, -V0) impurity pairs
E, Mxx, Mxy = solve_config(m, imp, 0.1; xi=0.5)

Er = real.(E)
F  = hall_sumrule_curve(Mxy, Er)            # sigma_xy for every mu at once
curve_at(F, Er, 1.0) / m.Nk                 # -> 1.0, the nu = 1 plateau
```

A full disorder average, with every observable taken from one
diagonalisation per configuration:

```julia
run_disorder(nLL=5, L=20, nimp=3556, V0=0.1, xi=0.5, kind=:pm,
             nconfig=2000, seed0=0,
             mugrid      = collect(0.0:0.002:5.5),
             omegas      = [0.0, 0.02, 0.05, 0.10],
             delta       = 0.001,
             spec_omegas = 0.0:0.002:2.0,
             spec_mus    = [0.62, 0.90, 1.35],
             outdir      = "output")
```

Run it threaded: `julia -t 64 --gcthreads=8`. Independent runs (different
`seed0`) combine with `merge_avg`.

See [`examples/`](examples/).

## API

**Geometry.** `lattice(a,b,c,d)` builds the supercell matrix (`abs(det)` is
the number of flux quanta); `kpoints`, `g1g2`, `projector` are the
Brillouin-zone helpers.

**Model.** `LLModel(nLL, klat; M=2, a0, G1, G2)` precomputes the form
factors and phase tables. It depends only on the geometry, so build it once
and reuse it across configurations. Fields include `Nk` (= $N_\phi$), `dim`,
`nLL`.

**Disorder.** `gen_imp_pm(m, nimp)` places `nimp` positive and `nimp`
negative impurities; `gen_imp_rand(m, nimp)` places `nimp` of random fixed
sign. Both take an `rng` keyword, so a configuration is reproducible from
its seed. `Vtable(m, imp, V0; xi)` gives $V(q)$ on the whole momentum grid.

**Hamiltonian.** `build_H(m, imp, V0; xi, threaded)` returns a `Hermitian`
matrix. `build_H!` writes into caller-owned storage.

**One configuration.** `solve_config(m, imp, V0; xi)` returns
`(E, Mxx, Mxy)` with `Mxy[a,b] = <a|jx|b><b|jy|a>`. In a loop use
`LLWorkspace(m, nimp_total)` and `solve_config!(ws, ...)`, which reuses its
buffers — the returned matrices alias the workspace, so consume them before
the next call.

**Observables.** These return a *curve* indexed by occupation `p`, giving
every chemical potential from a single pass; read it off with
`curve_at(curve, E, mu)`.

| | |
|---|---|
| `hall_sumrule_curve(Mxy, E; cutoff)` | $\mathrm{Im}\sum M_{ab}/\Delta E^2$ |
| `sigma_curve(omega, M, E, eta)` | $\sigma(\omega,\mu)$ at one frequency |
| `sigma_spectrum(omegas, mus, M, E, eta; refine)` | $\sigma(\omega)$ on a uniform grid, for a few $\mu$ |
| `dos_curve(omegas, E, delta)` | $-\mathrm{Im}\sum 1/(\omega-E_a+i\delta)$ |

**Drivers.** `run_disorder`, `run_spectra`, `merge_avg`.

## Conventions and units

| quantity | convention |
|---|---|
| energy, frequency | $\hbar\omega_c$ |
| length | magnetic length $\ell_B$ |
| conductivity | divide `hall_sumrule` by `N_phi` (`m.Nk`) for $e^2/h$ |
| | divide `sigma_curve` / `sigma_spectrum` output by `2*N_phi` |

The factor of 2 is not arbitrary: taking $\omega\to0$, $\eta\to0$ in the
Kubo expression gives exactly twice the sum rule. Both routes are evaluated
in `examples/gap_crossing.jl` and agree to five digits on the plateaux.

`nimp` counts **pairs** for `kind=:pm`, so there are `2*nimp` scatterers.
`xi > 0` selects gaussian impurities of that range, `xi = 0` point
scatterers.

$\mathrm{Im}\,\sigma_{xy}(\omega=0) = 0$ identically: at $\omega=0$ the Kubo
expression reduces to
$\sum 2(\mathrm{Re}M\,\eta + \mathrm{Im}M\,\Delta E)/(\Delta E(\Delta E^2+\eta^2))$,
which is real. Any low-frequency structure lives strictly at $\omega>0$.

## Output

`run_disorder` writes a `.jld2` containing

| key | |
|---|---|
| `sigma_xy`, `sigma_xx` | complex, `[mu, omega]`, with `_err` companions |
| `hall_sumrule` | real, `[mu]`, with `hall_sumrule_err` |
| `dos` | real, `[mu]` |
| `spec_sigma_xy`, `spec_sigma_xx` | `[omega, mu]`, if `spec_omegas` was given |
| `mugrid`, `omegas`, `delta`, `Nk`, `dim`, `nconfig`, `meta` | |

It checkpoints every `batch` configurations, so partial results are readable
while a run is in progress.

## Tests

```julia
using Pkg; Pkg.test("DisorderLL")
```
