# DisorderLL

Disorder-averaged transport in Landau levels, by exact diagonalisation.

Solves

$$H = \hbar\omega_c\left(n+\tfrac12\right) + V(\mathbf r)$$

projected onto `nLL` Landau levels on a magnetic-Brillouin-zone torus, with
randomly placed impurities, and computes the Kubo conductivities
$\sigma_{xx}(\omega,\mu)$, $\sigma_{xy}(\omega,\mu)$ and the Hall (TKNN) sum
rule, averaged over disorder realisations.

The package is a rewrite of a working but slow direct implementation.  It is
**1000×–4000× faster** on the Hamiltonian build and ~10³× faster on the
chemical-potential sweeps, while reproducing the original numbers to machine
precision (see [Correctness](#correctness)).

---

## Install

```julia
using Pkg
Pkg.develop(path="/path/to/disorder_LLs")   # or Pkg.add(url=...)
```

Julia ≥ 1.9.  Dependencies: `Distributions`, `JLD2`, `FFTW` (optional — the
spectral routine falls back to direct convolution if it is unavailable).

## Quick start

```julia
using DisorderLL

m   = LLModel(5, lattice(20,0,0,20))       # 5 Landau levels, 20x20 supercell
imp = gen_imp_pm(m, 3556)                   # 3556 (+V0, -V0) impurity pairs
E, Mxx, Mxy = solve_config(m, imp, 0.1; xi=0.5)

Er = real.(E)
F  = hall_sumrule_curve(Mxy, Er)            # sigma_xy for EVERY mu at once
curve_at(F, Er, 1.0) / m.Nk                 # -> 1.0, the nu=1 plateau
```

A full disorder average, with all observables from one diagonalisation per
configuration:

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

Output is a `.jld2` with `sigma_xy`, `sigma_xx` (complex, `[mu, omega]`, with
`_err` companions), `hall_sumrule`, `dos`, `spec_sigma_xy`, and metadata.
Independent runs (different `seed0`) combine with `merge_avg`.

See [`examples/`](examples/).

## Conventions and units

| quantity | convention |
|---|---|
| energy, frequency | $\hbar\omega_c$ |
| length | magnetic length $\ell_B$ |
| conductivity | divide `hall_sumrule` by `N_phi` (= `m.Nk`) for $e^2/h$ |
| | divide `sigma_curve` / `sigma_spectrum` output by `2*N_phi` |

The factor of 2 is not a fudge: taking $\omega\to0$, $\eta\to0$ in the Kubo
expression gives exactly twice the sum rule.  Both routes are computed in
`examples/gap_crossing.jl` and agree to four decimals on the plateaux, which
is how the normalisation should be checked rather than asserted.

`kind=:pm` places `nimp` positive **and** `nimp` negative impurities (so
`nimp` counts *pairs*, `2*nimp` scatterers); `kind=:rand` places `nimp`
impurities of random fixed sign.  `xi > 0` selects gaussian impurities of that
range, `xi = 0` point scatterers.

`Im σ_xy(ω=0) = 0` identically — at $\omega=0$ the Kubo expression reduces to
$\sum 2(\mathrm{Re}M\,\eta + \mathrm{Im}M\,\Delta E)/(\Delta E(\Delta E^2+\eta^2))$,
which is real.  Any low-frequency structure lives strictly at $\omega>0$.

## Where the speed comes from

1. **Hamiltonian.**  $V(q)$ and the Landau-level form factor depend only on
   the integer label $q = k_1-k_2+b$, not on $(i,j)$ separately, so both are
   tabulated once per configuration.  The impurity sum becomes a single
   `zgemm` using the separability
   $e^{-i\mathbf q\cdot\mathbf r} = A_r^{q_1}B_r^{q_2}$.  The remaining
   phases are $\eta_b\,e^{i(c_g/2)(b\times k)}$ lookup tables.  The direct
   version evaluated $\dim^2/2\times(2M+1)^2\times n_{\rm imp}$ complex
   exponentials, plus as many Laguerre-polynomial constructions and
   Brillouin-zone folds, *per configuration*.

2. **Chemical-potential sweeps.**  Every observable has the form
   $F(\mu) = \sum_{a\le p<b} T_{ab}$ with $T$ **antisymmetric**, hence
   $F(p) = F(p-1) + \sum_b T_{pb}$.  One row-sum pass plus a cumulative sum
   gives *every* $\mu$ at once: $O(\dim^2)$ in total instead of per $\mu$.

3. **Frequency spectra.**  $\sigma(\omega)$ is a convolution of the
   $\Delta E$-resolved spectral weight with $1/(x+i\eta)$; the weight is
   binned and convolved with one cached FFT plan —
   $O(\dim^2 + N\log N)$ instead of $O(N_\omega\dim^2)$.

4. **Currents.**  $J$ is block-bidiagonal in the Landau index and diagonal in
   $k$, so $JU$ costs $O(\dim^2)$; only $U^\dagger(JU)$ needs a `gemm`.

5. **Allocation.**  `LLWorkspace` + `solve_config!` reuse every large array
   except the LAPACK eigenvector matrix (a naive configuration at
   $\dim=2000$ allocates ~110 MB, which with 64 threads puts the run in the
   garbage collector).

6. **Threading.**  BLAS is pinned to one thread while configurations run in
   parallel, and accumulators are owned by a spawned worker for its lifetime
   rather than indexed by `threadid()`, which is not migration-safe under
   Julia's dynamic scheduler.  Run with `julia -t N --gcthreads=8`.

### Measured

| | direct | this package | |
|---|---|---|---|
| `H`, L=8, nLL=4, nimp=500 | 42.0 s | 0.0095 s | **4400×** |
| `H`, L=15, nLL=4, nimp=2000 | hours | 0.11 s | |
| Hall sum rule, 3501 μ (dim=900) | 4.59 s | 0.0043 s | **1060×** |
| σ_xy(ω,μ), 1501 ω × 30 μ | 22.1 s | 0.38 s | **58×** |
| full configuration, dim=2000 | | 15.3 s serial | incl. `eigen` |

Parallel scaling saturates near ~12× on 64 threads; that is the LAPACK
ceiling, not the package — concurrent single-threaded `eigen` calls on
OpenBLAS give only 13.7× on the same machine.

## Correctness

`test/runtests.jl` (428 assertions) checks the optimised routines against
**naive reimplementations of the original formulas** — direct double loops
over occupied/empty pairs, one μ at a time — plus:

* clean limit: exactly `nLL` levels at $n+\frac12$, each `N_phi`-fold degenerate
* Hermiticity of `H`, of the current matrices and of `M`
* binned+FFT spectra against the exact pair sum
* workspace path identical to the allocating path, and reusable
* quantised Hall plateaux at $\nu e^2/h$
* $\mathrm{Im}\,\sigma_{xy}(0)=0$; DOS normalisation $\int\!\rho\,dE/\pi = \dim$

```julia
using Pkg; Pkg.test("DisorderLL")
```

## Convergence — read this before quoting numbers

Three separate convergence axes, all of which matter:

* **Broadening η.**  Results are *not* automatically η-independent.  The
  plateau sets in below η ≈ 0.001–0.002 $\hbar\omega_c$; at η = 0.005 the
  low-frequency $\mathrm{Im}\,\sigma_{xy}$ is still 15–30 % below its
  $\eta\to0$ value.  The criterion is
  $\text{ensemble level spacing} \ll \eta \ll$ scale of variation of
  $\langle\sigma\rangle$, and the lower bound is met by the *ensemble*
  (width/$N_\phi N_{\rm cfg}$), not by any single sample.
* **Sample count N.**  The statistical error scales as
  $\eta^{-1/2}N^{-1/2}$: small η is expensive.  Reaching 1 % needs
  $N\sim10^3$ at η = 0.005 but $N\sim3\times10^4$ at η = 5×10⁻⁴.  At N = 25
  and η = 5×10⁻⁴ the answer can come out with the **wrong sign**.
* **System size.**  At fixed *filling factor* (not fixed μ — the levels are
  skewed) L = 15, 20, 25 agree within error bars, so L = 20 is already in the
  thermodynamic limit for these quantities.

A sensible production point: **η = 0.001, N ≈ 2000, L = 20**.

## Physics notes

* The disorder-broadened levels **narrow with Landau index** (widths 0.535,
  0.405, 0.350, 0.325 at $V_0=0.1$, $\xi=0.5\ell_B$).  This is a *finite-range*
  effect, not a consequence of the wavefunction being larger: for point
  scatterers all levels have identical width, by Laguerre orthonormality,
  $\int_0^\infty L_n(x)^2e^{-x}dx = 1$.  With range ξ the variance is
  $\sigma_n^2\propto\int_0^\infty L_n(x)^2e^{-sx}dx$, $s = 1+2\xi^2$, which
  predicts the measured ratios to 2–5 %.
* The DOS is **asymmetric even with equal numbers of $+V_0$ and $-V_0$
  impurities**.  Within one level $H$ is linear in $V$ and the ensemble is
  invariant under $V\to-V$, so a single-level DOS is exactly symmetric
  (verified: skewness $=-0.006\pm0.008$, mean exactly $0.5$).  Level mixing
  enters at second order, $\propto V^2/\hbar\omega_c$, which is *even* in $V$
  and so survives the $\pm$ cancellation.  The measured shift scales as
  $V_0^2$ to 1 %.

## Provenance

`src/lattice.jl` (`lattice`, `kpoints`, `g1g2`, `projector`) is taken verbatim
from `EDfunctions_square.jl` in the original project so that the magnetic
Brillouin-zone conventions match exactly.  Everything else is a rewrite.

Two bugs in the original implementation are **not** reproduced here:

* `V_rand(q, V0, imp_loc)` re-drew each impurity's sign *inside* the
  momentum loop, so the sign was re-randomised for every matrix element and
  the resulting `H` was not the Fourier transform of any potential.
  `gen_imp_rand` draws the signs once per configuration.
* `generate_imp` used `klattice[1,2]` where `klattice[2,1]` belongs when
  forming the second supercell vector (harmless for a square supercell, wrong
  in general).  `supercell` samples the actual parallelogram.
