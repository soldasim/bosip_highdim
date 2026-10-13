include("pilot.jl")

const FAILED_METHODS = Dict("dsp_map" => DimScaledMAP(), "saas_map_a0.1" => SaasMAP())

"Re-samples a run from its last saved parameters with lengthscales clamped at `LENGTHSCALE_CAP` widths."
function resample_clamped(p::LatentProblem, method::Method, seed, ref; sample_count=2000)
    dir = run_dir(p, method, seed)
    X, Y, params = load(dir * "_iters.jld2", "X", "Y", "params")
    Random.seed!(seed)
    (; model, acquisition, sampler) = setup(method, p)
    bosip = BosipProblem(ExperimentData(X, Y);
        f = simulator(p), domain = Domain(; bounds = bounds(p)), acquisition, model,
        likelihood = likelihood_model(p), x_prior = prior(p),
    )
    lb, ub = bounds(p)
    update_parameters!(bosip.problem, FixedParams(clamp_lengthscales(get_params(params[end]), LENGTHSCALE_CAP * (ub .- lb))))
    samples = resample(sample_expected_posterior(bosip, sampler, sample_count)..., sample_count)
    jldsave(dir * "_samples.jld2"; samples, clamped = true)
    return samples
end

"Replaces the NaN metrics of failed runs in `file` by those of the clamped re-sampling."
function resample_failed(file)
    results = load(file, "results")
    for (i, r) in enumerate(results)
        (isnan(r.metrics.sliced_wasserstein) && haskey(FAILED_METHODS, r.key[2])) || continue
        name, mname, seed, _ = r.key
        p = get_problem(replace(name, r"_d\d+(_aligned)?$" => ""), 10; rotate = !endswith(name, "_aligned"))
        ref, _ = cached_reference(p)
        samples = try
            resample_clamped(p, FAILED_METHODS[mname], seed, ref)
        catch e
            @warn "Re-sampling failed" r.key exception = e
            continue
        end
        results[i] = (; r..., metrics = evaluate(p, samples, ref), clamped = true)
        jldsave(file; results)
        println("done ", r.key)
    end
end
