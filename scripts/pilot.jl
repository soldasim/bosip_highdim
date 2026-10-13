include("../src/include_code.jl")
LinearAlgebra.BLAS.set_num_threads(1) # avoid oversubscription with Julia threads

get_problem(name, d; rotate) = PROBLEMS[name](d; rotate)

lengthscale_summary(p::LatentProblem, bosip) = lengthscale_summary(p, bosip.problem.model, bosip.problem.params.params)

function lengthscale_summary(p::LatentProblem, ::Any, params::GaussianProcessParams)
    λ = params.λ[:, 1]
    return (; λ_informed = median(λ[1:p.k]), λ_other = isempty(λ[p.k+1:end]) ? NaN : median(λ[p.k+1:end]), overlap = NaN)
end

"Squared cosine of the principal angles between the learned subspace and the true informed subspace, averaged."
function subspace_overlap(p::LatentProblem, E::AbstractMatrix)
    U = Matrix(qr(E').Q)[:, 1:min(size(E, 1), p.d)]
    V = Matrix(qr(p.Q[1:p.k, :]').Q)[:, 1:p.k]
    return sum(abs2, V' * U) / p.k
end

function lengthscale_summary(p::LatentProblem, model::LatentGaussianProcess, params::LatentGaussianProcessParams)
    return (; λ_informed = NaN, λ_other = NaN, overlap = mean(subspace_overlap(p, encoder(model, w)) for w in eachcol(params.W)))
end

"""
Run all combinations and append results to `file`.
`problems` is a vector of `(name, rotate)`. Posterior scores are recorded every `score_every` iterations.
"""
function pilot(; problems, methods, seeds, d=10, budget=200, n_init=20, score_every=5, file="data/pilot_curves.jld2")
    results = isfile(file) ? load(file, "results") : Any[]
    for (name, rotate) in problems, method in methods, seed in seeds
        p = get_problem(name, d; rotate)
        key = (p.name, method_name(method), seed, budget)
        any(r -> r.key == key, results) && continue
        ref, _ = cached_reference(p)
        res = try
            @timed run_experiment(p, method; seed, budget, n_init, ref, score_every)
        catch e
            e isa InterruptException && rethrow()
            @warn "Run failed" key exception = e
            open(replace(file, ".jld2" => "_failures.txt"), "a") do io
                println(io, key, "  ", sprint(showerror, e)[1:min(end, 200)])
            end
            continue
        end
        t, res = res.time, res.value
        metrics = isnothing(res.samples) ? (; mode_recall = NaN, sliced_wasserstein = NaN) : evaluate(p, res.samples, ref)
        r = (; key, time = t, metrics, scores = res.scores, ls = lengthscale_summary(p, res.bosip))
        push!(results, r)
        mkpath(dirname(file))
        jldsave(file; results)
        @info "DONE" key time=round(t) metrics ls = r.ls
    end
    return results
end
