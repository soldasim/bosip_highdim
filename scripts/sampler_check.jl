# AMIS on the true posterior (no surrogate): isolates sampler error.
include("../src/include_code.jl")

batched_loglike(p) = x -> ndims(x) == 1 ? only(true_loglike(p, reshape(x, :, 1))) : true_loglike(p, x)

function sampler_check(d=10; count=2000)
    sampler = setup(Baseline(), multimodal_problem(2)).sampler
    for (_, make) in sort(collect(PROBLEMS); by=first)
        p = make(d)
        ref, ref_ess = cached_reference(p)
        t = @elapsed xs, ws = sample_posterior(sampler, batched_loglike(p), prior(p), Domain(; bounds = bounds(p)), count)
        samples = resample(xs, ws, count)
        ess = sum(ws)^2 / sum(abs2, ws)
        @info "$(p.name)" time=round(t; digits=1) ref_ess=round(ref_ess) amis_ess=round(ess) evaluate(p, samples, ref)...
    end
end
