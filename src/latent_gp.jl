import BOSS: mean, var, cov, mean_and_var, mean_and_cov

"""
    LatentGaussianProcess(; kernel, latent_dim, projection, weight_priors, amplitude_priors, noise_std_priors, max_lengthscale)

A GP on the latent inputs `z = W * projection * x`, where the `latent_dim × m` matrix `W` is fitted
as a model parameter (one per output dimension). The latent lengthscales are fixed to one.
If `max_lengthscale` is finite, `x / max_lengthscale` is appended to `z`, so that all effective lengthscales are at most `max_lengthscale`.

## Keywords
- `projection::AbstractMatrix`: Fixed `m × d` map applied before `W`. Use the identity for a fully learned encoder.
- `weight_priors`: Per-output priors over the vectorized `W` (length `latent_dim * m`).
"""
struct LatentGaussianProcess{K<:Kernel} <: SurrogateModel
    kernel::K
    latent_dim::Int
    projection::Matrix{Float64}
    floor::Float64
    weight_priors::Vector{<:MultivariateDistribution}
    amplitude_priors::BOSS.AmplitudePriors
    noise_std_priors::BOSS.NoiseStdPriors
end
LatentGaussianProcess(; kernel=Matern52Kernel(), latent_dim, projection, weight_priors, amplitude_priors, noise_std_priors, max_lengthscale=Inf) =
    LatentGaussianProcess(kernel, latent_dim, Matrix{Float64}(projection), 1 / max_lengthscale, weight_priors, amplitude_priors, noise_std_priors)

"Parameters of the [`LatentGaussianProcess`](@ref); `W` stores the vectorized encoder matrices as columns."
struct LatentGaussianProcessParams{
    W<:AbstractMatrix{<:Real},
    A<:AbstractVector{<:Real},
    N<:AbstractVector{<:Real},
} <: ModelParams{LatentGaussianProcess}
    W::W
    α::A
    σ::N
end

"The learned `latent_dim × d` matrix mapping `x` to the latent inputs."
encoder(model::LatentGaussianProcess, w::AbstractVector{<:Real}) =
    reshape(w, model.latent_dim, size(model.projection, 1)) * model.projection

"The learned encoder with the rows `I / max_lengthscale` appended (if any)."
metric_encoder(model::LatentGaussianProcess, w::AbstractVector{<:Real}) =
    iszero(model.floor) ? encoder(model, w) : vcat(encoder(model, w), model.floor * I(size(model.projection, 2)))

BOSS.make_discrete(model::LatentGaussianProcess, discrete) = model
BOSS.sliceable(::Type{<:LatentGaussianProcess}) = true
BOSS.dimension_independent_given_parameters(::Type{<:LatentGaussianProcess}) = true

BOSS.slice(m::LatentGaussianProcess, idx::Int) = LatentGaussianProcess(
    m.kernel, m.latent_dim, m.projection, m.floor, m.weight_priors[idx:idx], m.amplitude_priors[idx:idx], m.noise_std_priors[idx:idx],
)
BOSS.slice(p::LatentGaussianProcessParams, idx::Int) =
    LatentGaussianProcessParams(p.W[:, idx:idx], p.α[idx:idx], p.σ[idx:idx])
BOSS.join_slices(ps::AbstractVector{<:LatentGaussianProcessParams}) =
    LatentGaussianProcessParams(reduce(hcat, getfield.(ps, :W)), reduce(vcat, getfield.(ps, :α)), reduce(vcat, getfield.(ps, :σ)))

latent_param_priors(model::LatentGaussianProcess) = vcat(model.weight_priors, model.amplitude_priors, model.noise_std_priors)

function BOSS.data_loglike(model::LatentGaussianProcess, data::ExperimentData)
    ll_slice(w, y, α, σ) = (E = metric_encoder(model, w); BOSS.gp_data_loglike_slice(E * data.X, y, nothing, model.kernel, ones(size(E, 1)), α, σ))
    return params -> sum(ll_slice.(eachcol(params.W), eachrow(data.Y), params.α, params.σ))
end

BOSS.params_logprior(model::LatentGaussianProcess) = params -> (
    sum(logpdf.(model.weight_priors, eachcol(params.W))) +
    sum(logpdf.(model.amplitude_priors, params.α)) +
    sum(logpdf.(model.noise_std_priors, params.σ))
)

BOSS._params_sampler(model::LatentGaussianProcess) = rng -> LatentGaussianProcessParams(
    reduce(hcat, rand.(Ref(rng), model.weight_priors)),
    rand.(Ref(rng), model.amplitude_priors),
    rand.(Ref(rng), model.noise_std_priors),
)

function BOSS.vectorizer(model::LatentGaussianProcess)
    is_dirac, dirac_vals = BOSS.create_dirac_mask(latent_param_priors(model))

    vectorize(params::LatentGaussianProcessParams) =
        BOSS.filter_diracs(vcat(vec(params.W), params.α, params.σ), is_dirac)

    function devectorize(params::LatentGaussianProcessParams, ps::AbstractVector{<:Real})
        ps = BOSS.insert_diracs(ps, is_dirac, dirac_vals)
        w_len, α_len = length(params.W), length(params.α)
        return LatentGaussianProcessParams(
            reshape(ps[1:w_len], size(params.W)),
            ps[w_len+1:w_len+α_len],
            ps[w_len+α_len+1:end],
        )
    end

    return vectorize, devectorize
end

BOSS.bijector(model::LatentGaussianProcess) = BOSS.simplify(BOSS.default_bijector(latent_param_priors(model)))

"Posterior of a single output; delegates to the GP posterior on the latent inputs."
struct LatentGaussianProcessPosterior{P<:GaussianProcessPosterior} <: ModelPosteriorSlice{LatentGaussianProcess}
    post::P
    encoder::Matrix{Float64}
end

function BOSS.model_posterior_slice(model::LatentGaussianProcess, params::LatentGaussianProcessParams, data::ExperimentData, slice::Int)
    E = metric_encoder(model, params.W[:, slice])
    α, σ, y = params.α[slice], params.σ[slice], data.Y[slice, :]
    post_gp = BOSS._posdef_retry(α; context = "latent posterior_gp, slice $slice") do jitter
        fgp = BOSS.finite_gp(E * data.X, nothing, model.kernel, ones(size(E, 1)), α, sqrt(σ^2 + jitter))
        AbstractGPs.posterior(fgp, y)
    end
    return LatentGaussianProcessPosterior(GaussianProcessPosterior(post_gp), E)
end

mean(p::LatentGaussianProcessPosterior, x::AbstractVecOrMat{<:Real}) = mean(p.post, p.encoder * x)
var(p::LatentGaussianProcessPosterior, x::AbstractVecOrMat{<:Real}) = var(p.post, p.encoder * x)
cov(p::LatentGaussianProcessPosterior, X::AbstractMatrix{<:Real}) = cov(p.post, p.encoder * X)
mean_and_var(p::LatentGaussianProcessPosterior, x::AbstractVecOrMat{<:Real}) = mean_and_var(p.post, p.encoder * x)
mean_and_cov(p::LatentGaussianProcessPosterior, X::AbstractMatrix{<:Real}) = mean_and_cov(p.post, p.encoder * X)
