abstract type Method end

"Default BOSIP setup (matches the `bosip_benchmarks` defaults)."
struct Baseline <: Method end

"Baseline with the SAAS lengthscale prior, fitted by MAP."
struct SaasMAP <: Method
    α::Float64
end
SaasMAP(; α=0.1) = SaasMAP(α)

"""
Baseline with the dimension-scaled lengthscale prior (Hvarfner et al., 2024), fitted by MAP.
Lengthscales are `LogNormal(μ₀ + log(d)/2, σ₀)` in units of the domain width.
"""
struct DimScaledMAP <: Method
    μ₀::Float64
    σ₀::Float64
end
DimScaledMAP(; μ₀=sqrt(2), σ₀=sqrt(3)) = DimScaledMAP(μ₀, σ₀)

"Baseline with the SAAS lengthscale prior, fitted by NUTS sampling."
struct SaasBI <: Method
    α::Float64
    chain_count::Int
    warmup::Int
    samples_in_chain::Int
    leap_size::Int
end
SaasBI(; α=0.1, chain_count=4, warmup=64, samples_in_chain=4, leap_size=2) =
    SaasBI(α, chain_count, warmup, samples_in_chain, leap_size)

"""
GP on a learned linear encoder `z = W x` with `k` latent dimensions (`k = nothing` uses the true number of informed coordinates).
Fitted by MAP.
"""
struct LatentGP <: Method
    k::Union{Nothing, Int}
end
LatentGP(; k=3) = LatentGP(k)

"Oracle: GP on the true informed subspace `Q[1:k, :] x`, with a learned `k × k` linear map. Fitted by MAP."
struct OracleGP <: Method end

"Wraps `method` and truncates its lengthscale priors at `cap` domain widths."
struct Capped{M<:Method} <: Method
    method::M
    cap::Float64
end

method_name(::Baseline) = "baseline"
method_name(m::Capped) = "$(method_name(m.method))_cap$(m.cap)"
method_name(m::LatentGP) = "latent_k$(something(m.k, "true"))"
method_name(::OracleGP) = "oracle_gp"
method_name(::DimScaledMAP) = "dsp_map"
method_name(m::SaasMAP) = "saas_map_a$(m.α)"
method_name(m::SaasBI) = "saas_bi_a$(m.α)_c$(m.chain_count)w$(m.warmup)"

lengthscale_priors(::Baseline, p::LatentProblem) = gp_priors(p).lengthscale_priors
function lengthscale_priors(m::Union{SaasMAP, SaasBI}, p::LatentProblem)
    lb, ub = bounds(p)
    return fill(SaasPrior(ub .- lb; α = m.α), y_size(p))
end

noise_std_priors(::Method, p::LatentProblem) = gp_priors(p).noise_std_priors
noise_std_priors(::SaasBI, p::LatentProblem) = fill(Dirac(1e-4), y_size(p)) # `Dirac(0.)` has a non-finite gradient under AD

function lengthscale_priors(m::DimScaledMAP, p::LatentProblem)
    lb, ub = bounds(p)
    μ = m.μ₀ + log(p.d) / 2 .+ log.(ub .- lb)
    return fill(product_distribution(LogNormal.(μ, m.σ₀)), y_size(p))
end

function lengthscale_priors(m::Capped, p::LatentProblem)
    lb, ub = bounds(p)
    return truncate_prior.(lengthscale_priors(m.method, p), Ref(m.cap .* (ub .- lb)))
end
truncate_prior(d::Product, ub) = product_distribution(truncated.(d.v, 0., ub))
truncate_prior(d::SaasPrior, ub) = SaasPrior(d.scales; α = d.α, grid = d.log_τ, ub)

noise_std_priors(m::Capped, p::LatentProblem) = noise_std_priors(m.method, p)
model_fitter(m::Capped) = model_fitter(m.method)

model_fitter(::Method) = OptimizationMAP(;
    algorithm = NEWUOA(), multistart = 2, warm_start = true, parallel = true, rhoend = 1e-4,
)
model_fitter(m::SaasBI) = TuringBI(;
    sampler = NUTS(m.warmup, 0.65), m.warmup, m.samples_in_chain, m.chain_count, m.leap_size, parallel = true,
)

amis_sampler(multistart) = AMISSampler(;
    iters = 10,
    proposal_fitter = BOSIP.AnalyticalFitter(),
    gauss_mix_options = GaussMixOptions(;
        algorithm = BOBYQA(), multistart, parallel = true,
        cluster_ϵs = nothing, rel_min_weight = 1e-8, rhoend = 1e-4,
    ),
)

function surrogate_model(method::Method, p::LatentProblem)
    priors = (; gp_priors(p)..., lengthscale_priors = lengthscale_priors(method, p), noise_std_priors = noise_std_priors(method, p))
    return GaussianProcess(; kernel = Matern52Kernel(), priors...)
end

latent_dim(m::LatentGP, p::LatentProblem) = something(m.k, p.k)
latent_dim(::OracleGP, p::LatentProblem) = p.k
projection(::LatentGP, p::LatentProblem) = Matrix(1.0I, p.d, p.d)
projection(::OracleGP, p::LatentProblem) = p.Q[1:p.k, :]

function latent_model(method::Union{LatentGP, OracleGP}, p::LatentProblem)
    k, P = latent_dim(method, p), projection(method, p)
    width = 2 * HALF_WIDTH
    weight_prior = product_distribution(fill(Normal(0., 3 / width), k * size(P, 1)))
    priors = gp_priors(p)
    return LatentGaussianProcess(;
        latent_dim = k, projection = P,
        weight_priors = fill(weight_prior, y_size(p)),
        priors.amplitude_priors, priors.noise_std_priors,
    )
end

"Baseline GP with the additive kernel over consecutive coordinate pairs; the true structure of the summed problems."
function additive_model(p::LatentProblem)
    blocks = [collect(i:i+1) for i in 1:2:p.d]
    kernel = (1 / length(blocks)) * sum(Matern52Kernel() ∘ SelectTransform(b) for b in blocks)
    return GaussianProcess(; kernel, gp_priors(p)...)
end

surrogate_model(method::LatentGP, p::LatentProblem) = latent_model(method, p)
surrogate_model(method::OracleGP, p::LatentProblem) = p.additive ? additive_model(p) : latent_model(method, p)

function setup(method::Method, p::LatentProblem)
    return (;
        model = surrogate_model(method, p),
        acquisition = LogMaxVar(),
        model_fitter = model_fitter(method),
        acq_maximizer = OptimizationAM(;
            algorithm = BOBYQA(), multistart = 2, parallel = true, rhoend = 1e-4,
        ),
        sampler = amis_sampler(24),
        score_sampler = amis_sampler(8),
    )
end
