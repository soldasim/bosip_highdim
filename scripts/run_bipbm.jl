# usage: julia --project=. scripts/run_bipbm.jl <bipbm-problem> [budget=200] [seed=1] [method=baseline] [n_init=20] [d=10] [rotate=true] [score=true] [save=true]
#        julia --project=. scripts/run_bipbm.jl list
# Methods: baseline, dsp, saas (all with lengthscale priors truncated at 5 domain widths), linenc (linear encoder, k=3).
# Problems need a Gaussian likelihood and a uniform prior on the bounds (or `[-5, 5]^d` coordinates).
include("../src/include_code.jl")

const RUN_METHODS = Dict(
    "baseline" => () -> Capped(Baseline(), 5.),
    "dsp" => () -> Capped(DimScaledMAP(), 5.),
    "saas" => () -> Capped(SaasMAP(), 5.),
    "linenc" => () -> LinearEncoder(; k=3),
)

parse_option(v) = (x = tryparse(Int, v); !isnothing(x) ? x : v in ("true", "false") ? v == "true" : v)

"""
    run_bipbm(id; method="baseline", budget=200, seed=1, n_init=20, score=true, save=true, sample_count=2000, problem=(;))

Run BOSIP on the BIPBM problem `id` and return the experiment result.
If `score`, the posterior is scored (sliced Wasserstein-1) every 5 iterations against samples of the BIPBM ground truth.
`problem` holds keywords of the BIPBM constructor, e.g. `(; d = 20, rotate = false)`.
"""
function run_bipbm(id; method="baseline", budget=200, seed=1, n_init=20, score=true, save=true, sample_count=2000, problem=(;))
    bip = BIPBM.problem(id; problem...)
    p = bipbm_problem(bip; rng = MersenneTwister(seed))
    ref = nothing
    if score
        ref = BIPBM.posterior_samples(bip; n = sample_count, rng = MersenneTwister(seed))
        isnothing(p.box) || (ref = to_unit(ref, p.box...))
    end
    @info "Problem" id name = p.name d = p.d y = y_size(p) budget method
    res = run_experiment(p, RUN_METHODS[method](); seed, budget, n_init, sample_count, ref, save)
    isnothing(res.samples) || isnothing(ref) || @info "Final" evaluate(p, res.samples, ref)...
    return (; p, bip, res)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if first(ARGS) == "list"
        foreach(println, BIPBM.problem_names())
    else
        opts = Dict(Symbol(k) => parse_option(v) for (k, v) in split.(ARGS[2:end], "="))
        problem = (; (k => pop!(opts, k) for k in (:d, :rotate) if haskey(opts, k))...)
        run_bipbm(ARGS[1]; problem, opts...)
    end
end
