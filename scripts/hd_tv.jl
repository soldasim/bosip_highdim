include("../src/include_code.jl")

"Total-variation scores of a stored run, recomputed from the saved data and hyperparameters of every `every`-th iteration."
function tv_scores(p::LatentProblem, method::Method, seed; every=5, n_init=3, file=run_dir(p, method, seed) * "_iters.jld2")
    grid = load(joinpath("grids", p.name, "posterior_grid.jld2"))
    metric = TVMetric(; grid = to_unit(grid["xs"], p.box...), log_ws = grid["log_ws"], true_logvals = grid["true_logvals"])
    X, Y, params = load(file, "X", "Y", "params")
    (; model, acquisition) = setup(method, p)

    iterations = 1:every:length(params)
    scores = map(iterations) do j
        n = n_init + j - 1
        bosip = BosipProblem(ExperimentData(X[:, 1:n], Y[:, 1:n]);
            f = simulator(p), domain = Domain(; bounds = bounds(p)), acquisition, model, likelihood = likelihood_model(p), x_prior = prior(p))
        update_parameters!(bosip.problem, FixedParams(params[j].params))
        tv = try
            calculate_metric(metric, _ -> NaN, log_posterior_mean(bosip))
        catch e
            e isa InterruptException && rethrow()
            @warn "TV failed" j exception = e
            NaN
        end
        @info "TV" j n tv
        (; n, tv)
    end
    return scores
end

"""
Checks that the grid of `p` agrees with the true log-posterior of our (rescaled) problem up to a constant, on `count` grid points.
"""
function check_grid(p::LatentProblem; count=300, rng=MersenneTwister(1))
    grid = load(joinpath("grids", p.name, "posterior_grid.jld2"))
    idx = randperm(rng, size(grid["xs"], 2))[1:count]
    V = to_unit(grid["xs"][:, idx], p.box...)
    ours = true_loglike(p, V) .+ logpdf(prior(p), V)
    diff = grid["true_logvals"][idx] .- ours
    return (; offset = mean(diff), spread = std(diff), finite = mean(isfinite.(diff)))
end

function hd_tv(name, method; seeds=1:3, every=5, check=true)
    p = PROBLEMS[name](5)
    check && @info "Grid check" name check_grid(p)
    for seed in seeds
        out = run_dir(p, method, seed) * "_tv.jld2"
        isfile(out) && continue
        scores = tv_scores(p, method, seed; every)
        jldsave(out; scores)
    end
end
