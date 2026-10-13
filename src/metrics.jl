"""
Reference posterior samples by importance sampling from the prior.
Returns `(samples, ess)`; a small `ess` means the reference is unreliable.
"""
function reference_samples(p::LatentProblem; count=2000, chunk=10^6, chunks=10, rng=Random.default_rng())
    pr = prior(p)
    X = reduce(hcat, (rand(rng, pr, chunk) for _ in 1:chunks))
    logw = true_loglike(p, X)
    w = exp.(logw .- maximum(logw))
    ess = sum(w)^2 / sum(abs2, w)
    return resample(X, w, count), ess
end

function cached_reference(p::LatentProblem; dir="data", kwargs...)
    file = joinpath(dir, "reference_$(p.name).jld2")
    isfile(file) && return load(file, "samples", "ess")
    samples, ess = reference_samples(p; kwargs...)
    mkpath(dir)
    jldsave(file; samples, ess)
    return samples, ess
end

"Fraction of true modes holding at least `min_mass` of the samples (informed coords)."
function mode_recall(p::LatentProblem, X::AbstractMatrix; min_mass=0.01)
    size(p.mode_centers, 2) == 0 && return NaN
    V = informed(p, X)
    found = map(eachcol(p.mode_centers)) do c
        mean(sqrt.(sum(abs2, V .- c; dims=1)) .< p.mode_radius) >= min_mass
    end
    return mean(found)
end

"Sliced Wasserstein-1 distance between equal-size sample sets (informed coords)."
function sliced_wasserstein(p::LatentProblem, X::AbstractMatrix, Y::AbstractMatrix; projections=200, rng=Random.default_rng())
    n = min(size(X, 2), size(Y, 2))
    U, V = informed(p, X)[:, 1:n], informed(p, Y)[:, 1:n]
    dirs = normalize.(eachcol(randn(rng, p.k, projections)))
    return mean(θ -> mean(abs.(sort(vec(θ' * U)) .- sort(vec(θ' * V)))), dirs)
end

function evaluate(p::LatentProblem, X::AbstractMatrix, ref::AbstractMatrix)
    return (;
        mode_recall = mode_recall(p, X),
        sliced_wasserstein = sliced_wasserstein(p, X, ref),
    )
end
