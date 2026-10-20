include("evidence_study.jl")
import OptimizationPRIMA
const PRIMA = OptimizationPRIMA.PRIMA

const ML_VARIANTS = [
    ("ab_emb", true), ("ab_emb", false), ("ab_sum", false),
    ("simple_emb", true), ("simple_emb", false), ("simple_sum", false),
    ("banana_emb", true), ("banana_emb", false), ("banana_sum", false),
    ("bimodal_emb", true), ("bimodal_emb", false), ("bimodal_sum", false),
    ("wavy_ridge", true), ("wavy_ridge", false),
    ("wavy_multimodal", true), ("wavy_multimodal", false),
]
const ML_FILE = "data/map_study.jld2"

"Maximises `f` from each start with NEWUOA; returns the best `(value, θ)`."
function maximise(f, starts; rhobeg=0.2, rhoend=1e-4, maxfun=10_000)
    best = (-Inf, first(starts))
    for x0 in starts
        x, _ = PRIMA.newuoa(θ -> (v = f(θ); isfinite(v) ? -v : 1e10), x0; rhobeg, rhoend, maxfun)
        v = f(x)
        v > best[1] && (best = (v, x))
    end
    return best
end

"Adds a small random row to the encoder of `θ` (nested start for `k + 1`)."
function pad_encoder(rng, θ, k, d)
    W = reshape(θ[1:k*d], k, d)
    return vcat(vec(vcat(W, 0.05 .* randn(rng, 1, d))), θ[end])
end

"""
MAP fit of `(W, α)` (maximum of data log-likelihood plus log-prior in α, with the capped metric) for `k = 1:d`, with nested warm starts.
The optimisation runs over `log α` without the Jacobian term, so it equals the mode in `α`. The fitted `θ = [vec(W); log α]` is saved.
"""
function ml_job(job)
    (; name, rotate, design, seed, n, output) = job
    p = get_problem(name, 10; rotate)
    X, Y = DATA_CACHE[(name, rotate, design, seed)]
    X, Y = X[:, 1:n], Y[:, 1:n]
    rng = MersenneTwister(hash((p.name, design, seed, n, output)))
    out, prev = Any[], nothing
    for k in 1:p.d
        t = SMCTarget(p, k, X, Y; output)
        starts = isnothing(prev) ? [t.sample_prior(rng), t.sample_prior(rng)] : [pad_encoder(rng, prev, k - 1, p.d), t.sample_prior(rng)]
        logprior(θ) = t.logprior(θ) - θ[end]
        time = @elapsed (logpost, θ) = maximise(θ -> t.loglike(θ) + logprior(θ), starts)
        prev = θ
        push!(out, (; job..., problem = p.name, k, logpost, ll = t.loglike(θ), logprior = logprior(θ), θ, time))
    end
    return out
end

"Runs `ml_job` over all variants, outputs, designs and values of `n`; `workers` tasks pull jobs from a queue."
function ml_study(; variants=ML_VARIANTS, designs=[:A, :B], seeds=[1], ns=STUDY_NS, workers=2, file=ML_FILE)
    results = isfile(file) ? load(file, "results") : Any[]
    done = Set((r.name, r.rotate, r.design, r.seed, r.n, r.output) for r in results)
    jobs = [(; name, rotate, design, seed, n, output) for design in designs for (name, rotate) in variants for seed in seeds for n in ns
        for output in 1:y_size(get_problem(name, 10; rotate))]
    jobs = sort(filter(j -> (j.name, j.rotate, j.design, j.seed, j.n, j.output) ∉ done, jobs); by = j -> j.n)
    cache_data!(jobs)
    @info "ML study" jobs = length(jobs) workers
    queue = Channel{Any}(length(jobs))
    foreach(j -> put!(queue, j), jobs)
    close(queue)
    lock = ReentrantLock()
    tasks = [Threads.@spawn(for job in queue
        r = try
            ml_job(job)
        catch e
            e isa InterruptException && rethrow()
            @warn "ML job failed" job exception = e
            continue
        end
        Base.@lock lock begin
            append!(results, r)
            jldsave(file; results)
            @info "DONE" r[1].problem job.design job.n job.output logpost = round.([x.logpost for x in r]; digits = 1)
        end
    end) for _ in 1:workers]
    foreach(wait, tasks)
    return results
end
