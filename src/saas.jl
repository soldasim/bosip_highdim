using Bijectors

"""
    SaasPrior(scales; α=0.1)

Sparse axis-aligned subspace prior (Eriksson & Jankowiak, 2021) over the lengthscales `ℓ`
with the global shrinkage `τ ~ HalfCauchy(α)` integrated out numerically.

With `ℓ̃ = ℓ ./ scales`: `ρᵢ = ℓ̃ᵢ⁻² | τ ~ HalfCauchy(τ)`.

## Keywords
- `scales`: Domain widths, so that `ℓ̃` lives in the unit cube.
- `α`: Scale of the global shrinkage prior.
- `grid`: Quadrature grid over `log τ`.
- `ub`: Upper bounds on `ℓ` (the prior is truncated there, up to normalization).
"""
struct SaasPrior{T<:Real} <: ContinuousMultivariateDistribution
    scales::Vector{T}
    α::T
    log_τ::Vector{Float64}
    ub::Vector{T}
end
SaasPrior(scales::AbstractVector{<:Real}; α=0.1, grid=range(-14, 14; length=561), ub=fill(Inf, length(scales))) =
    SaasPrior(collect(float.(scales)), float(α), collect(float.(grid)), collect(float.(ub)))

Base.length(d::SaasPrior) = length(d.scales)
Base.eltype(::SaasPrior{T}) where {T} = T
Distributions.insupport(d::SaasPrior, x::AbstractVector{<:Real}) = all(>(0), x) && all(x .<= d.ub)

logsumexp(x) = (m = maximum(x); m + log(sum(v -> exp(v - m), x)))
log_halfcauchy(x, s) = log(2 / π) - log(s) - log1p((x / s)^2)

function Distributions._logpdf(d::SaasPrior, ℓ::AbstractVector{<:Real})
    any(ℓ .> d.ub) && return -Inf
    ℓ̃ = ℓ ./ d.scales
    ρ = ℓ̃ .^ -2
    ll_τ = map(d.log_τ) do u
        τ = exp(u)
        log_halfcauchy(τ, d.α) + u + sum(r -> log_halfcauchy(r, τ), ρ)
    end
    log_ρ = logsumexp(ll_τ) + log(d.log_τ[2] - d.log_τ[1])
    return log_ρ + sum(log(2) .- 3 .* log.(ℓ̃)) - sum(log, d.scales)
end

function Distributions._rand!(rng::AbstractRNG, d::SaasPrior, x::AbstractVector{<:Real})
    τ = d.α * abs(rand(rng, Cauchy()))
    ρ = τ .* abs.(rand(rng, Cauchy(), length(d)))
    x .= clamp.(d.scales ./ sqrt.(ρ), 1e-3 .* d.scales, min.(1e3 .* d.scales, d.ub))
end

Bijectors.bijector(d::SaasPrior) = all(isfinite, d.ub) ?
    Bijectors.Stacked(Bijectors.Logit.(0., d.ub), [i:i for i in eachindex(d.ub)]) :
    Bijectors.elementwise(log)
