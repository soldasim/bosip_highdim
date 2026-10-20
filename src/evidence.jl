"""
    SMCTarget

Posterior over the parameters `θ = [vec(W); log α]` of one output of a [`LatentGaussianProcess`](@ref) with `W ∈ ℝ^{k×d}`.
Built by `SMCTarget(p, k, X, Y)`.
"""
struct SMCTarget{L,P,S}
    k::Int
    d::Int
    loglike::L
    logprior::P
    sample_prior::S
end

"Result of [`smc_evidence`](@ref); `particles` is `(k*d + 1) × P`."
struct SMCResult
    logZ::Float64
    particles::Matrix{Float64}
    loglikes::Vector{Float64}
    temperatures::Int
    acceptance::Float64
end

"The target for output `Y[output, :]` of the capped linear encoder with `k` latent dimensions."
function SMCTarget(p::LatentProblem, k::Int, X::AbstractMatrix, Y::AbstractMatrix; output::Int=1)
    model = BOSS.slice(latent_model(LinearEncoder(; k), p), output)
    ll = BOSS.data_loglike(model, ExperimentData(X, Y[output:output, :]))
    d = p.d
    wp, amp = model.weight_priors[1], model.amplitude_priors[1]
    params(θ) = LatentGaussianProcessParams(reshape(θ[1:k*d], :, 1), [exp(θ[end])], [0.])

    function loglike(θ)
        v = try
            ll(params(θ))
        catch e
            e isa InterruptException && rethrow()
            -Inf
        end
        return isfinite(v) ? v : -Inf
    end
    logprior(θ) = logpdf(wp, θ[1:k*d]) + logpdf(amp, exp(θ[end])) + θ[end]
    sample_prior(rng) = vcat(rand(rng, wp), log(rand(rng, amp)))
    return SMCTarget(k, d, loglike, logprior, sample_prior)
end

logsumexp(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))

"Largest `β′ ∈ (β, 1]` such that the effective sample size of the incremental weights stays above `ess_target`."
function next_temperature(ll, β, ess_target)
    ess(β′) = (lw = (β′ - β) .* ll; exp(2 * logsumexp(lw) - logsumexp(2 .* lw)))
    ess(1.) >= ess_target && return 1.
    lo, hi = β, 1.
    for _ in 1:60
        mid = (lo + hi) / 2
        ess(mid) >= ess_target ? (lo = mid) : (hi = mid)
    end
    return max(lo, nextfloat(β))
end

function systematic_resample(rng, w)
    n = length(w)
    cw = cumsum(w)
    cw[end] = 1.
    return searchsortedfirst.(Ref(cw), (rand(rng) .+ (0:n-1)) ./ n)
end

"Multiplies `W` of every particle by a Haar-random orthogonal matrix; prior and likelihood are invariant to this."
function rotate_encoders!(rng, θs, k, d)
    for θ in eachcol(θs)
        Q, R = qr(randn(rng, k, k))
        W = reshape(view(θ, 1:k*d), k, d)
        W .= (Matrix(Q) * Diagonal(sign.(diag(R)))) * W
    end
end

"Per-parameter proposal scales: the particle std of each input column of `W` (pooled over rows), and of `log α`."
function proposal_scales(θs, k, d)
    Wp = reshape(view(θs, 1:k*d, :), k, d, size(θs, 2))
    s = vcat(repeat(vec(std(Wp; dims = (1, 3))), inner = k), std(view(θs, size(θs, 1), :)))
    return max.(s, 1e-3)
end

function metropolis_sweep!(rng, θs, ll, lp, t::SMCTarget, β, scale, c)
    accepted = 0
    for i in axes(θs, 2)
        θ′ = θs[:, i] .+ c .* scale .* randn(rng, size(θs, 1))
        lp′ = t.logprior(θ′)
        isfinite(lp′) || continue
        ll′ = t.loglike(θ′)
        isfinite(ll′) || continue
        if log(rand(rng)) < β * (ll′ - ll[i]) + lp′ - lp[i]
            θs[:, i] .= θ′
            ll[i], lp[i] = ll′, lp′
            accepted += 1
        end
    end
    return accepted / size(θs, 2)
end

"""
    smc_evidence(rng, target; particles, moves, ess_frac)

Tempered SMC from the prior to the posterior with adaptive temperatures.
Each step reweights and resamples, applies an exact random rotation of every `W`, then `moves` random-walk Metropolis sweeps.
Returns the log-evidence estimate and the final particles.
"""
function smc_evidence(rng, t::SMCTarget; particles=100, moves=10, ess_frac=0.5, max_temperatures=2000)
    P, D = particles, t.k * t.d + 1
    θs = reduce(hcat, [t.sample_prior(rng) for _ in 1:P])
    ll = [t.loglike(θ) for θ in eachcol(θs)]
    lp = [t.logprior(θ) for θ in eachcol(θs)]
    any(isfinite, ll) || error("All prior samples have non-finite likelihood.")
    β, logZ, steps, acc_sum, c = 0., 0., 0, 0., 2.38 / sqrt(D)

    while β < 1
        steps += 1
        steps > max_temperatures && error("SMC exceeded $max_temperatures temperatures.")
        β′ = next_temperature(ll, β, ess_frac * P)
        lw = (β′ - β) .* ll
        logZ += logsumexp(lw) - log(P)
        idx = systematic_resample(rng, exp.(lw .- logsumexp(lw)))
        θs, ll, lp = θs[:, idx], ll[idx], lp[idx]
        β = β′

        rotate_encoders!(rng, θs, t.k, t.d)
        scale = proposal_scales(θs, t.k, t.d)
        for _ in 1:moves
            acc = metropolis_sweep!(rng, θs, ll, lp, t, β, scale, c)
            acc_sum += acc / moves
            c = clamp(c * exp(0.5 * (acc - 0.25)), 1e-3, 3.)
        end
    end
    return SMCResult(logZ, θs, ll, steps, acc_sum / steps)
end
