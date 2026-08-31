# A converged disorder average: DOS, Hall sum rule, sigma_xx, and the
# frequency spectra -- all from one diagonalisation per configuration.
#
#   julia -t 64 --gcthreads=8 --project=. examples/disorder_average.jl
#
# Settings are the converged ones (see README, "Convergence"):
#   eta = 0.001 on the broadening plateau, N = 2000, L = 20.
using DisorderLL

run_disorder(
    nLL = 5, L = 20, nimp = 3556,       # impurity density 8.889 pairs/flux quantum
    V0 = 0.1, xi = 0.5, kind = :pm,
    nconfig = parse(Int, get(ENV, "NCONFIG", "2000")), seed0 = 0,

    mugrid    = collect(0.0:0.002:5.5),
    omegas    = [0.0, 0.02, 0.05, 0.10],
    delta     = 0.001,
    dos_delta = 0.005,

    spec_omegas = 0.0:0.002:2.0,
    spec_mus    = [0.62, 0.90, 1.35],   # above n=0 peak / gap / below n=1 peak
    spec_refine = 8,

    batch = 100,
    outdir = get(ENV, "OUTDIR", joinpath(pwd(), "output")),
    label = "example",
)
