"Log-likelihood of the simulator at every column of `X`, evaluated in parallel; failures give `-Inf`."
function threaded_loglike(p::LatentProblem, X::AbstractMatrix)
    ll = Vector{Float64}(undef, size(X, 2))
    Threads.@threads :greedy for i in axes(X, 2)
        v = try
            true_loglike(p, X[:, i:i])[1]
        catch e
            e isa InterruptException && rethrow()
            -Inf
        end
        ll[i] = isfinite(v) ? v : -Inf
    end
    return ll
end

"""
    smc_reference(p; particles, ess_frac, moved, max_sweeps, final_sweeps, count, rng)

Samples the true posterior of `p` by tempered SMC with adaptive temperatures and differential-evolution Metropolis moves.
Every temperature is followed by sweeps until each particle has moved `moved` times on average.
Returns `(samples, info)` with `count` samples.
"""
function smc_reference(p::LatentProblem; particles=4000, ess_frac=0.5, moved=2., max_sweeps=50, final_sweeps=20, count=2000, rng=Random.default_rng())
    pr = prior(p)
    d = p.d
    X = rand(rng, pr, particles)
    ll = threaded_loglike(p, X)
    lp = logpdf(pr, X)
    β, steps, evals, c = 0., 0, particles, 1.
    acc_sum, sweep_count = 0., 0

    function sweep!(β)
        a, b = rand(rng, 1:particles, particles), rand(rng, 1:particles, particles)
        Xp = X .+ c * 2.38 / sqrt(2d) .* (X[:, a] .- X[:, b]) .+ 1e-6 .* randn(rng, d, particles)
        lpp = logpdf(pr, Xp)
        llp = fill(-Inf, particles)
        ok = isfinite.(lpp)
        llp[ok] = threaded_loglike(p, Xp[:, ok])
        evals += sum(ok)
        accept = log.(rand(rng, particles)) .< β .* (llp .- ll) .+ lpp .- lp
        X[:, accept], ll[accept], lp[accept] = Xp[:, accept], llp[accept], lpp[accept]
        acc = mean(accept)
        c = clamp(c * exp(0.5 * (acc - 0.25)), 1e-2, 3.)
        acc_sum += acc
        sweep_count += 1
        return acc
    end

    while β < 1
        steps += 1
        β′ = next_temperature(ll, β, ess_frac * particles)
        lw = (β′ - β) .* ll
        idx = systematic_resample(rng, exp.(lw .- logsumexp(lw)))
        X, ll, lp = X[:, idx], ll[idx], lp[idx]
        β = β′
        total, sweeps = 0., 0
        while total < moved && sweeps < max_sweeps
            total += sweep!(β)
            sweeps += 1
        end
    end
    foreach(_ -> sweep!(1.), 1:final_sweeps)

    samples = X[:, randperm(rng, particles)[1:min(count, particles)]]
    info = (; temperatures = steps, evaluations = evals, acceptance = acc_sum / sweep_count, unique = length(unique(eachcol(X))) / particles)
    return samples, info
end
