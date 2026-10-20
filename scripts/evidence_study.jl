include("../src/include_code.jl")
LinearAlgebra.BLAS.set_num_threads(1)

const STUDY_PROBLEMS = [("ab_emb", true), ("ab_emb", false), ("ab_sum", false)]
const STUDY_NS = [20, 50, 100, 200]
const STUDY_FILE = "data/evidence_study.jld2"

get_problem(name, d; rotate) = PROBLEMS[name](d; rotate)

"The number of relevant directions of the (single) output."
reference_k(p::LatentProblem) = p.additive ? p.d : p.k

"200 points: prior draws (design `:A`) or the data of a finished DS-GP run (design `:B`)."
function study_data(p::LatentProblem, design::Symbol, seed::Int)
    if design == :A
        X = rand(MersenneTwister(10_000 + seed), prior(p), 200)
        return X, reduce(hcat, simulator(p).(eachcol(X)))
    else
        return load(joinpath("data", p.name, "dsp_map_cap5.0_$(seed)_iters.jld2"), "X", "Y")
    end
end

function study_test_set(p::LatentProblem; count=1000)
    X = rand(MersenneTwister(999), prior(p), count)
    return X, vec(reduce(hcat, simulator(p).(eachcol(X))))
end

"Held-out RMSE (relative to the std of `ytest`) of the best particle and of the ensemble mean of `count` particles."
function predictive_errors(p::LatentProblem, t::SMCTarget, res::SMCResult, X, Y, Xtest, ytest; count=30)
    k, d = t.k, t.d
    model = latent_model(LinearEncoder(; k), p)
    data = ExperimentData(X, Y[1:1, :])
    function predict(θ)
        params = LatentGaussianProcessParams(reshape(θ[1:k*d], :, 1), [exp(θ[end])], [0.])
        try
            return vec(mean(BOSS.model_posterior_slice(model, params, data, 1), Xtest))
        catch e
            e isa InterruptException && rethrow()
            return nothing
        end
    end
    rel_rmse(μ) = isnothing(μ) ? NaN : sqrt(mean(abs2, μ .- ytest)) / std(ytest)
    logpost = res.loglikes .+ t.logprior.(eachcol(res.particles))
    best = rel_rmse(predict(res.particles[:, argmax(logpost)]))
    μs = filter(!isnothing, predict.(eachcol(res.particles[:, 1:min(count, end)])))
    return (; rmse_best = best, rmse_ens = isempty(μs) ? NaN : rel_rmse(mean(μs)))
end

const DATA_CACHE = Dict{Any, Any}()

"Loads each dataset once, sequentially (JLD2 is not thread safe)."
function cache_data!(jobs)
    for (name, rotate, design, seed) in unique((j.name, j.rotate, j.design, j.seed) for j in jobs)
        get!(DATA_CACHE, (name, rotate, design, seed)) do
            study_data(get_problem(name, 10; rotate), design, seed)
        end
    end
end

function run_job(job; particles=100, moves=150, ess_frac=0.9)
    (; name, rotate, design, seed, n, k) = job
    p = get_problem(name, 10; rotate)
    X, Y = DATA_CACHE[(name, rotate, design, seed)]
    X, Y = X[:, 1:n], Y[:, 1:n]
    t = SMCTarget(p, k, X, Y)
    rng = MersenneTwister(hash((p.name, design, seed, n, k)))
    time = @elapsed res = smc_evidence(rng, t; particles, moves, ess_frac)
    Xtest, ytest = study_test_set(p)
    errs = predictive_errors(p, t, res, X, Y, Xtest, ytest)
    return (; job..., problem = p.name, k_ref = reference_k(p), res.logZ, res.temperatures, res.acceptance, time, errs...)
end

"Runs all jobs in parallel over the threads and appends to `file`; finished jobs are skipped."
function evidence_study(; problems=STUDY_PROBLEMS, designs=[:A, :B], seeds=1:3, ns=STUDY_NS, ks=n -> n == 200 ? (1:4) : (1:10), particles=100, moves=150, ess_frac=0.9, file=STUDY_FILE)
    results = isfile(file) ? load(file, "results") : Any[]
    key(r) = (r.name, r.rotate, r.design, r.seed, r.n, r.k)
    done = Set(key.(results))
    jobs = [(; name, rotate, design, seed, n, k) for (name, rotate) in problems for design in designs for seed in seeds for n in ns for k in ks(n)]
    jobs = sort(filter(j -> key(j) ∉ done, jobs); by = j -> (-j.n, -j.k))
    cache_data!(jobs)
    @info "Evidence study" jobs = length(jobs) threads = Threads.nthreads()
    lock = ReentrantLock()
    Threads.@threads :greedy for job in jobs
        r = try
            run_job(job; particles, moves, ess_frac)
        catch e
            e isa InterruptException && rethrow()
            @warn "Job failed" job exception = e
            continue
        end
        Base.@lock lock begin
            push!(results, r)
            mkpath(dirname(file))
            jldsave(file; results)
            @info "DONE" r.problem r.design r.seed r.n r.k logZ = round(r.logZ; digits = 2) T = r.temperatures acc = round(r.acceptance; digits = 2) time = round(r.time; digits = 1) rmse_ens = round(r.rmse_ens; digits = 3)
        end
    end
    return results
end

"Independent SMC replicates of the log-evidence on one dataset, to measure the estimator noise."
function evidence_replicates(; name="ab_emb", rotate=true, design=:A, seed=1, ns=[50, 100], ks=[1, 2, 3, 5, 10], reps=4, particles=100, moves=10)
    p = get_problem(name, 10; rotate)
    X, Y = study_data(p, design, seed)
    jobs = [(; n, k, rep) for n in ns for k in ks for rep in 1:reps]
    out = Vector{Any}(undef, length(jobs))
    Threads.@threads :dynamic for i in eachindex(jobs)
        (; n, k, rep) = jobs[i]
        t = SMCTarget(p, k, X[:, 1:n], Y[:, 1:n])
        res = smc_evidence(MersenneTwister(hash((n, k, rep))), t; particles, moves)
        out[i] = (; n, k, rep, logZ = res.logZ, T = res.temperatures, acc = res.acceptance)
    end
    for n in ns, k in ks
        z = [o.logZ for o in out if o.n == n && o.k == k]
        println("REPL n=$n k=$k logZ mean=", round(mean(z); digits = 2), " std=", round(std(z); digits = 2), " T=", [o.T for o in out if o.n == n && o.k == k][1], " all=", round.(z; digits = 1))
    end
    return out
end

"Log-evidence for different SMC settings on one dataset, to find converged settings."
function evidence_convergence(; name="ab_emb", rotate=true, design=:A, seed=1, n=100, ks=[1, 2], reps=3,
    configs=[(400, 0.5, 20), (1600, 0.5, 20), (400, 0.9, 20), (400, 0.9, 60), (1600, 0.9, 20)])
    p = get_problem(name, 10; rotate)
    X, Y = study_data(p, design, seed)
    jobs = [(; k, cfg, rep) for k in ks for cfg in configs for rep in 1:reps]
    out = Vector{Any}(undef, length(jobs))
    Threads.@threads :dynamic for i in eachindex(jobs)
        (; k, cfg, rep) = jobs[i]
        t = SMCTarget(p, k, X[:, 1:n], Y[:, 1:n])
        time = @elapsed res = smc_evidence(MersenneTwister(hash((k, cfg, rep))), t; particles = cfg[1], ess_frac = cfg[2], moves = cfg[3])
        out[i] = (; k, cfg, rep, logZ = res.logZ, T = res.temperatures, acc = res.acceptance, time)
    end
    for k in ks, cfg in configs
        o = [x for x in out if x.k == k && x.cfg == cfg]
        z = [x.logZ for x in o]
        println("CONV n=$n k=$k P=$(cfg[1]) ess=$(cfg[2]) moves=$(cfg[3]) logZ=", round(mean(z); digits = 2), " std=", round(std(z); digits = 2), " T=", o[1].T, " acc=", round(o[1].acc; digits = 2), " time=", round(mean(x.time for x in o); digits = 0), "s")
    end
    return out
end
