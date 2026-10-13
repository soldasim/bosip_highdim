"Saves data and fitted model parameters after every iteration."
struct SaveCallback <: BosipCallback
    file::String
    params::Vector{Any}
end
SaveCallback(file) = SaveCallback(file, Any[])

function (cb::SaveCallback)(bosip::BosipProblem; kwargs...)
    push!(cb.params, deepcopy(bosip.problem.params))
    data = bosip.problem.data
    jldsave(cb.file; X = data.X, Y = data.Y, params = cb.params)
end

"Scores the posterior approximation against `ref` every `every`-th call. Scoring errors give `NaN`."
mutable struct ScoreCallback <: BosipCallback
    p::LatentProblem
    sampler::Any
    ref::Matrix{Float64}
    count::Int
    every::Int
    file::Union{Nothing, String}
    calls::Int
    scores::Vector{Any}
end
ScoreCallback(p, sampler, ref; count=2000, every=1, file=nothing) =
    ScoreCallback(p, sampler, ref, count, every, file, 0, Any[])

function score_posterior(p::LatentProblem, bosip::BosipProblem, sampler, ref, count)
    try
        xs, ws = sample_expected_posterior(bosip, sampler, count; options = BosipOptions(; info = false))
        return evaluate(p, resample(xs, ws, count), ref)
    catch e
        e isa InterruptException && rethrow()
        @warn "Scoring failed" exception = e
        return (; mode_recall = NaN, sliced_wasserstein = NaN)
    end
end

function (cb::ScoreCallback)(bosip::BosipProblem; kwargs...)
    cb.calls += 1
    (cb.calls - 1) % cb.every == 0 || return
    n = size(bosip.problem.data.X, 2)
    push!(cb.scores, (; n, metrics = score_posterior(cb.p, bosip, cb.sampler, cb.ref, cb.count)))
    isnothing(cb.file) || jldsave(cb.file; scores = cb.scores)
end

run_dir(p, method, seed) = joinpath("data", p.name, "$(method_name(method))_$(seed)")

"""
    run_experiment(p, method; seed, budget, n_init, sample_count, ref, score_every)

Run BOSIP with `budget` simulations in total (including `n_init`).
If reference samples `ref` are given, the posterior is scored every `score_every` iterations.
Returns the final `BosipProblem`, posterior samples and the score history.
"""
function run_experiment(p::LatentProblem, method::Method=Baseline();
    seed = 1,
    budget = 200,
    n_init = 20,
    sample_count = 2000,
    ref = nothing,
    score_every = 5,
    save = true,
)
    rng = MersenneTwister(seed)
    Random.seed!(seed)
    (; model, acquisition, model_fitter, acq_maximizer, sampler, score_sampler) = setup(method, p)

    f = simulator(p)
    X = rand(rng, prior(p), n_init)
    Y = reduce(hcat, f.(eachcol(X)))
    bosip = BosipProblem(ExperimentData(X, Y);
        f,
        domain = Domain(; bounds = bounds(p)),
        acquisition,
        model,
        likelihood = likelihood_model(p),
        x_prior = prior(p),
    )

    dir = run_dir(p, method, seed)
    save && mkpath(dirname(dir))
    callbacks = BosipCallback[]
    save && push!(callbacks, SaveCallback(dir * "_iters.jld2"))
    score_cb = isnothing(ref) ? nothing : ScoreCallback(p, score_sampler, ref; count = sample_count, every = score_every,
        file = save ? dir * "_scores.jld2" : nothing)
    isnothing(score_cb) || push!(callbacks, score_cb)

    bosip!(bosip;
        model_fitter, acq_maximizer,
        term_cond = DataLimit(budget),
        options = BosipOptions(; callback = BOSIP.CombinedCallback(callbacks...), info = true),
    )

    samples, clamped = final_samples(bosip, sampler, sample_count)
    save && !isnothing(samples) && jldsave(dir * "_samples.jld2"; samples, clamped)
    return (; bosip, samples, clamped, scores = isnothing(score_cb) ? Any[] : score_cb.scores)
end

"Lengthscales above this many domain widths are clamped if the final sampling fails."
const LENGTHSCALE_CAP = 5.

clamp_lengthscales(params::GaussianProcessParams, cap) = GaussianProcessParams(min.(params.λ, cap), params.α, params.σ, params.μ)
clamp_lengthscales(params, cap) = nothing

"""
Samples the expected posterior. If that fails, the lengthscales are clamped at `LENGTHSCALE_CAP` domain widths and sampling is retried.
Returns `(samples, clamped)`; `samples` is `nothing` if sampling is impossible.
"""
function final_samples(bosip::BosipProblem, sampler, count)
    sample() = resample(sample_expected_posterior(bosip, sampler, count)..., count)
    try
        return sample(), false
    catch e
        e isa InterruptException && rethrow()
        @warn "Final sampling failed; clamping lengthscales" exception = e
    end
    lb, ub = bosip.problem.domain.bounds
    clamped = clamp_lengthscales(get_params(bosip.problem.params), LENGTHSCALE_CAP * (ub .- lb))
    isnothing(clamped) && return nothing, false
    update_parameters!(bosip.problem, FixedParams(clamped))
    try
        return sample(), true
    catch e
        e isa InterruptException && rethrow()
        @warn "Final sampling failed after clamping" exception = e
        return nothing, true
    end
end
