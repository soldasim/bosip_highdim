"""
    ProjectedGaussianProcess(gp, P)

An ordinary [`GaussianProcess`](@ref) `gp` on the inputs `P * x` for a fixed matrix `P`.
Has the same parameters as `gp` (`P.size[1]` ARD lengthscales per output).
"""
struct ProjectedGaussianProcess{G<:GaussianProcess} <: SurrogateModel
    gp::G
    P::Matrix{Float64}
end

"Parameters of the [`ProjectedGaussianProcess`](@ref); wraps the parameters of the inner GP."
struct ProjectedGaussianProcessParams{P<:GaussianProcessParams} <: ModelParams{ProjectedGaussianProcess}
    gp::P
end

project(m::ProjectedGaussianProcess, data::ExperimentData) = ExperimentData(m.P * data.X, data.Y)

BOSS.make_discrete(m::ProjectedGaussianProcess, discrete) = m
BOSS.sliceable(::Type{<:ProjectedGaussianProcess}) = true
BOSS.dimension_independent_given_parameters(::Type{<:ProjectedGaussianProcess}) = true
BOSS.slice(m::ProjectedGaussianProcess, idx::Int) = ProjectedGaussianProcess(BOSS.slice(m.gp, idx), m.P)

BOSS.slice(p::ProjectedGaussianProcessParams, idx::Int) = ProjectedGaussianProcessParams(BOSS.slice(p.gp, idx))
BOSS.join_slices(ps::AbstractVector{<:ProjectedGaussianProcessParams}) = ProjectedGaussianProcessParams(BOSS.join_slices(getfield.(ps, :gp)))

BOSS.data_loglike(m::ProjectedGaussianProcess, data::ExperimentData) =
    (ll = BOSS.data_loglike(m.gp, project(m, data)); params -> ll(params.gp))
BOSS.params_logprior(m::ProjectedGaussianProcess) =
    (lp = BOSS.params_logprior(m.gp); params -> lp(params.gp))
BOSS._params_sampler(m::ProjectedGaussianProcess) =
    (sampler = BOSS._params_sampler(m.gp); rng -> ProjectedGaussianProcessParams(sampler(rng)))
BOSS.bijector(m::ProjectedGaussianProcess) = BOSS.bijector(m.gp)

function BOSS.vectorizer(m::ProjectedGaussianProcess)
    vectorize, devectorize = BOSS.vectorizer(m.gp)
    return params -> vectorize(params.gp), (params, ps) -> ProjectedGaussianProcessParams(devectorize(params.gp, ps))
end

"Posterior of a single output; delegates to the GP posterior on the projected inputs."
struct ProjectedGaussianProcessPosterior{P<:ModelPosteriorSlice} <: ModelPosteriorSlice{ProjectedGaussianProcess}
    post::P
    P::Matrix{Float64}
end

BOSS.model_posterior_slice(m::ProjectedGaussianProcess, params::ProjectedGaussianProcessParams, data::ExperimentData, slice::Int) =
    ProjectedGaussianProcessPosterior(BOSS.model_posterior_slice(m.gp, params.gp, project(m, data), slice), m.P)

mean(p::ProjectedGaussianProcessPosterior, x::AbstractVecOrMat{<:Real}) = mean(p.post, p.P * x)
var(p::ProjectedGaussianProcessPosterior, x::AbstractVecOrMat{<:Real}) = var(p.post, p.P * x)
cov(p::ProjectedGaussianProcessPosterior, X::AbstractMatrix{<:Real}) = cov(p.post, p.P * X)
mean_and_var(p::ProjectedGaussianProcessPosterior, x::AbstractVecOrMat{<:Real}) = mean_and_var(p.post, p.P * x)
mean_and_cov(p::ProjectedGaussianProcessPosterior, X::AbstractMatrix{<:Real}) = mean_and_cov(p.post, p.P * X)
