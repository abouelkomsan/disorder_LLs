# ---------------------------------------------------------------------
# Production drivers.
# ---------------------------------------------------------------------

"""
Default output directory.  Override with the `outdir` keyword, or set the
environment variable `DISORDER_LL_OUTDIR` before loading the package.
"""
const SAVDIR = get(ENV, "DISORDER_LL_OUTDIR",
                   joinpath(pwd(), "disorder_LL_output"))


# ---------------------------------------------------------------------
# parallel driver: each spawned worker owns accumulator slot `w` for its
# whole lifetime, so task migration cannot corrupt it (indexing
# accumulators by `threadid()` is *not* safe under Julia's dynamic
# scheduler -- it can even segfault when threadid() exceeds nthreads()).
# ---------------------------------------------------------------------
function _parallel_configs(f, cfgrange, nworkers::Int)
    counter = Threads.Atomic{Int}(first(cfgrange))
    stop    = last(cfgrange)
    @sync for w in 1:nworkers
        Threads.@spawn begin
            while true
                ic = Threads.atomic_add!(counter, 1)
                ic > stop && break
                f(w, ic)
            end
        end
    end
end

"""
    tagof(; kwargs...) -> String

Canonical file-name stem for a run.
"""
function tagof(; kind, nLL, L, V0, xi, nimp)
    k = string(kind)
    x = xi == 0 ? "" : @sprintf("_xi_%g", xi)
    @sprintf("%s_nLL_%d_L_%d_V0_%g%s_nimp_%d", k, nLL, L, V0, x, nimp)
end

# ---------------------------------------------------------------------
"""
    run_disorder(; nLL, L, nimp, V0, xi=0.0, kind=:pm,
                   nconfig, seed0=0, M=2,
                   mugrid, omegas=[0.0], delta=1e-3,
                   cutoffs=Float64[], dos_delta=delta,
                   outdir=SAVDIR, save_eigen=false, batch=200)

Disorder-average sigma_xx(omega, mu) and sigma_xy(omega, mu) (complex: both
real and imaginary parts), the Hall sum rule Im sum M_xy/dE^2, and the DOS.

`kind`
  `:pm`   nimp positive + nimp negative impurities (nimp counts pairs)
  `:rand` nimp impurities of random sign +-V0, fixed per configuration
`xi > 0` switches to gaussian impurities of range `xi` (magnetic lengths).

Every observable is evaluated on the *whole* `mugrid` at the cost of one
O(dim^2) pass (see `hall_sumrule_curve` / `sigma_curve`), so a dense mu grid
is essentially free.

Results (running mean and variance over configurations) are checkpointed to
`outdir` every `batch` configurations.
"""
function run_disorder(; nLL::Int, L::Int, nimp::Int, V0::Real,
                        xi::Real=0.0, kind::Symbol=:pm,
                        nconfig::Int, seed0::Int=0, M::Int=2,
                        mugrid::AbstractVector{<:Real},
                        omegas::AbstractVector{<:Real}=[0.0],
                        delta::Real=1e-3,
                        cutoffs::AbstractVector{<:Real}=Float64[],
                        dos_delta::Real=delta,
                        spec_omegas::Union{Nothing,AbstractRange}=nothing,
                        spec_mus::AbstractVector{<:Real}=Float64[],
                        spec_refine::Int=4,
                        outdir::String=SAVDIR,
                        save_eigen::Bool=false,
                        batch::Int=200,
                        label::String="")

    mdl  = LLModel(nLL, lattice(L,0,0,L); M=M)
    nmu  = length(mugrid); nw = length(omegas); nc = length(cutoffs)
    tag  = tagof(; kind=kind, nLL=nLL, L=L, V0=V0, xi=xi, nimp=nimp)
    tag  = isempty(label) ? tag : label*"_"*tag
    mkpath(outdir)
    fout = joinpath(outdir, "avg_"*tag*".jld2")

    @printf("run_disorder: %s\n  dim=%d  Nk=%d  nconfig=%d  threads=%d\n",
            tag, mdl.dim, mdl.Nk, nconfig, Threads.nthreads())

    nt  = Threads.nthreads()
    nimp_tot = kind === :pm ? 2nimp : nimp
    wss = [LLWorkspace(mdl, nimp_tot) for _ in 1:nt]
    zc(dims...) = [zeros(ComplexF64, dims...) for _ in 1:nt]
    zr(dims...) = [zeros(Float64,    dims...) for _ in 1:nt]
    s_xy, s_xy2 = zc(nmu, nw), zr(nmu, nw)      # var of |.|
    s_xx, s_xx2 = zc(nmu, nw), zr(nmu, nw)
    s_sr, s_sr2 = zr(nmu),     zr(nmu)
    s_cut       = zr(nmu, max(nc,1))
    s_dos       = zr(nmu)
    done        = zeros(Int, nt)
    # full omega-spectra at a few mu, from the same diagonalisation
    do_spec = spec_omegas !== nothing && !isempty(spec_mus)
    nsw, nsm = do_spec ? (length(spec_omegas), length(spec_mus)) : (0,0)
    sp_xy = zc(max(nsw,1), max(nsm,1))
    sp_xx = zc(max(nsw,1), max(nsm,1))

    nblas = BLAS.get_num_threads(); BLAS.set_num_threads(1)
    t0 = time()
    try
        for b0 in 1:batch:nconfig
            b1 = min(b0+batch-1, nconfig)
            _parallel_configs(b0:b1, nt) do t, ic
                rng = Xoshiro(seed0 + ic)
                imp = kind === :pm ? gen_imp_pm(mdl, nimp; rng=rng) :
                                     gen_imp_rand(mdl, nimp; rng=rng)
                E, Mxx, Mxy = solve_config!(wss[t], mdl, imp, V0; xi=xi)
                Er = real.(E)

                F = hall_sumrule_curve(Mxy, Er)
                @inbounds for (i,mu) in enumerate(mugrid)
                    v = curve_at(F, Er, mu)
                    s_sr[t][i] += v;  s_sr2[t][i] += v*v
                end
                for (j,cut) in enumerate(cutoffs)
                    Fc = hall_sumrule_curve(Mxy, Er; cutoff=cut)
                    @inbounds for (i,mu) in enumerate(mugrid)
                        s_cut[t][i,j] += curve_at(Fc, Er, mu)
                    end
                end
                for (j,w) in enumerate(omegas)
                    cxy = sigma_curve(w, Mxy, Er, delta)
                    cxx = sigma_curve(w, Mxx, Er, delta)
                    @inbounds for (i,mu) in enumerate(mugrid)
                        vy = curve_at(cxy, Er, mu); vx = curve_at(cxx, Er, mu)
                        s_xy[t][i,j] += vy; s_xy2[t][i,j] += abs2(vy)
                        s_xx[t][i,j] += vx; s_xx2[t][i,j] += abs2(vx)
                    end
                end
                d = dos_curve(mugrid, Er, dos_delta)
                @inbounds for i in 1:nmu; s_dos[t][i] += d[i]; end

                if do_spec
                    sp_xy[t] .+= sigma_spectrum(spec_omegas, spec_mus, Mxy, Er, delta; refine=spec_refine)
                    sp_xx[t] .+= sigma_spectrum(spec_omegas, spec_mus, Mxx, Er, delta; refine=spec_refine)
                end

                if save_eigen
                    jldsave(joinpath(outdir, @sprintf("eig_%s_config_%d.jld2", tag, ic));
                            energies=Er, Mxx=Mxx, Mxy=Mxy)
                end
                done[t] += 1
            end
            n = sum(done)
            _save_avg(fout, mdl, mugrid, omegas, cutoffs, delta, dos_delta, n,
                      s_xy, s_xy2, s_xx, s_xx2, s_sr, s_sr2, s_cut, s_dos,
                      (; nLL, L, nimp, V0, xi, kind, seed0, M, tag),
                      do_spec, spec_omegas, spec_mus, sp_xy, sp_xx)
            @printf("  %6d / %d configs   %8.1f s   (%.2f s/config)\n",
                    n, nconfig, time()-t0, (time()-t0)/max(n,1))
            flush(stdout)
        end
    finally
        BLAS.set_num_threads(nblas)
    end
    println("  -> ", fout)
    return fout
end

function _save_avg(fout, mdl, mugrid, omegas, cutoffs, delta, dos_delta, n,
                   s_xy, s_xy2, s_xx, s_xx2, s_sr, s_sr2, s_cut, s_dos, meta,
                   do_spec=false, spec_omegas=nothing, spec_mus=Float64[],
                   sp_xy=nothing, sp_xx=nothing)
    red(v) = sum(v)
    xy, xx = red(s_xy)./n, red(s_xx)./n
    sr     = red(s_sr)./n
    xy2, xx2, sr2 = red(s_xy2)./n, red(s_xx2)./n, red(s_sr2)./n
    err(m2, m) = sqrt.(max.(m2 .- abs2.(m), 0) ./ max(n-1,1))
    jldsave(fout;
        nconfig    = n,
        mugrid     = collect(mugrid),
        omegas     = collect(omegas),
        cutoffs    = collect(cutoffs),
        delta      = delta,
        dos_delta  = dos_delta,
        Nk         = mdl.Nk,          # divide by this for sigma per flux quantum
        dim        = mdl.dim,
        sigma_xy   = xy,              # complex, [mu, omega]
        sigma_xx   = xx,
        sigma_xy_err = err(xy2, xy),
        sigma_xx_err = err(xx2, xx),
        hall_sumrule = sr,            # real,    [mu]
        hall_sumrule_err = err(sr2, sr),
        hall_sumrule_cutoff = red(s_cut)./n,
        dos        = red(s_dos)./n,
        spec_omegas = do_spec ? collect(spec_omegas) : Float64[],
        spec_mus    = do_spec ? collect(spec_mus)    : Float64[],
        spec_sigma_xy = do_spec ? red(sp_xy)./n : ComplexF64[;;],
        spec_sigma_xx = do_spec ? red(sp_xx)./n : ComplexF64[;;],
        meta       = meta,
        written    = string(now()))
end

# ---------------------------------------------------------------------
"""
    run_spectra(; ..., omegas, mus)

Disorder-averaged sigma_xx(omega) and sigma_xy(omega) on a dense uniform
`omegas` grid, for a handful of chemical potentials `mus`
(binned + FFT-convolved -- see `sigma_spectrum`).
"""
function run_spectra(; nLL::Int, L::Int, nimp::Int, V0::Real,
                       xi::Real=0.0, kind::Symbol=:pm,
                       nconfig::Int, seed0::Int=0, M::Int=2,
                       omegas::AbstractRange, mus::AbstractVector{<:Real},
                       delta::Real=1e-3, refine::Int=8,
                       outdir::String=SAVDIR, batch::Int=200, label::String="")

    mdl = LLModel(nLL, lattice(L,0,0,L); M=M)
    tag = tagof(; kind=kind, nLL=nLL, L=L, V0=V0, xi=xi, nimp=nimp)
    tag = isempty(label) ? tag : label*"_"*tag
    mkpath(outdir)
    fout = joinpath(outdir, @sprintf("spec_%s_eta_%g.jld2", tag, delta))
    nt = Threads.nthreads()
    wss = [LLWorkspace(mdl, kind === :pm ? 2nimp : nimp) for _ in 1:nt]
    acc_xy = [zeros(ComplexF64, length(omegas), length(mus)) for _ in 1:nt]
    acc_xx = [zeros(ComplexF64, length(omegas), length(mus)) for _ in 1:nt]
    done = zeros(Int, nt)

    @printf("run_spectra: %s  dim=%d  %d omegas x %d mus\n",
            tag, mdl.dim, length(omegas), length(mus))
    nblas = BLAS.get_num_threads(); BLAS.set_num_threads(1)
    t0 = time()
    try
        for b0 in 1:batch:nconfig
            b1 = min(b0+batch-1, nconfig)
            _parallel_configs(b0:b1, nt) do t, ic
                rng = Xoshiro(seed0 + ic)
                imp = kind === :pm ? gen_imp_pm(mdl, nimp; rng=rng) :
                                     gen_imp_rand(mdl, nimp; rng=rng)
                E, Mxx, Mxy = solve_config!(wss[t], mdl, imp, V0; xi=xi)
                Er = real.(E)
                acc_xy[t] .+= sigma_spectrum(omegas, mus, Mxy, Er, delta; refine=refine)
                acc_xx[t] .+= sigma_spectrum(omegas, mus, Mxx, Er, delta; refine=refine)
                done[t] += 1
            end
            n = sum(done)
            jldsave(fout; nconfig=n, omegas=collect(omegas), mus=collect(mus),
                    delta=delta, Nk=mdl.Nk, dim=mdl.dim,
                    sigma_xy=sum(acc_xy)./n, sigma_xx=sum(acc_xx)./n,
                    meta=(; nLL, L, nimp, V0, xi, kind, seed0, M, tag),
                    written=string(now()))
            @printf("  %6d / %d configs   %8.1f s\n", n, nconfig, time()-t0); flush(stdout)
        end
    finally
        BLAS.set_num_threads(nblas)
    end
    println("  -> ", fout)
    return fout
end

# ---------------------------------------------------------------------
"""
    merge_avg(out, files...)

Combine independent `run_disorder` outputs into one disorder average.

Each file stores means together with `nconfig`, so the combination is the
config-weighted mean  `(n1*m1 + n2*m2)/(n1+n2)`.  This is what makes it
unnecessary to decide the number of configurations up front: run a second
batch with a different `seed0` and merge.

Grids (`mugrid`, `omegas`, `spec_omegas`, `spec_mus`, `delta`, ...) must match.
"""
function merge_avg(out::String, files::AbstractString...)
    @assert length(files) >= 1
    ds = [load(f) for f in files]
    for k in ("mugrid","omegas","spec_omegas","spec_mus","cutoffs","delta","dos_delta","Nk","dim")
        ref = ds[1][k]
        for d in ds[2:end]
            @assert d[k] == ref "field `$k` differs between runs; refusing to merge"
        end
    end
    ns = [d["nconfig"] for d in ds]; N = sum(ns)
    wmean(k) = sum(n .* d[k] for (n,d) in zip(ns,ds)) ./ N
    # errors: recombine second moments, then re-derive the standard error
    function werr(mk, ek)
        m  = wmean(mk)
        m2 = sum(n .* (abs2.(d[mk]) .+ (max(d["nconfig"]-1,1)) .* abs2.(d[ek]))
                 for (n,d) in zip(ns,ds)) ./ N
        return sqrt.(max.(m2 .- abs2.(m), 0) ./ max(N-1,1))
    end
    merged = Dict{String,Any}()
    for k in ("mugrid","omegas","spec_omegas","spec_mus","cutoffs","delta","dos_delta","Nk","dim","meta")
        merged[k] = ds[1][k]
    end
    merged["nconfig"] = N
    for k in ("sigma_xy","sigma_xx","hall_sumrule","hall_sumrule_cutoff","dos",
              "spec_sigma_xy","spec_sigma_xx")
        haskey(ds[1], k) && (merged[k] = wmean(k))
    end
    merged["sigma_xy_err"]     = werr("sigma_xy","sigma_xy_err")
    merged["sigma_xx_err"]     = werr("sigma_xx","sigma_xx_err")
    merged["hall_sumrule_err"] = werr("hall_sumrule","hall_sumrule_err")
    merged["merged_from"]      = collect(files)
    merged["written"]          = string(now())
    jldsave(out; (Symbol(k)=>v for (k,v) in merged)...)
    @printf("merged %d runs (%s configs) -> %s\n", length(files), string(N), out)
    return out
end
