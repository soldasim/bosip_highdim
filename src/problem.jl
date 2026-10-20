"""
Benchmark whose likelihood depends only on the first `k` coordinates of the rotated input `Q * x`.
The remaining `d - k` directions are prior-dominated.

## Fields
- `g`: map from the `k` informed coordinates to the simulator output.
- `Q`: orthogonal `d × d` rotation.
- `mode_centers`: true posterior modes in informed coordinates (`k × n_modes`, may be empty).
- `mode_radius`: a sample within this distance (informed coords) of a center counts as that mode.
- `Σ_obs`: observation covariance; `nothing` means independent noise with `std_obs`.
- `additive`: the output is a sum over blocks of consecutive coordinates (all `d` coordinates informed).
- `x_prior`, `likelihood`, `loglike`: prior on `[-5, 5]^d`, BOSIP likelihood and true log-likelihood of the simulator output,
  if they differ from the defaults (truncated normal prior, Gaussian likelihood around `z_obs`).
- `box`: bounds `(lb, ub)` of the original parameters, which the coordinates `[-5, 5]^d` are rescaled from.
"""
struct LatentProblem{G}
    name::String
    d::Int
    k::Int
    g::G
    Q::Matrix{Float64}
    z_obs::Vector{Float64}
    std_obs::Vector{Float64}
    mode_centers::Matrix{Float64}
    mode_radius::Float64
    amp_est::Union{Float64, Vector{Float64}}
    Σ_obs::Union{Nothing, Matrix{Float64}}
    additive::Bool
    x_prior::Union{Nothing, Distribution}
    likelihood::Union{Nothing, Likelihood}
    loglike::Union{Nothing, Function}
    box::Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}}}
end

LatentProblem(name, d, k, g, Q, z_obs, std_obs, mode_centers, mode_radius, amp_est, Σ_obs, additive) =
    LatentProblem(name, d, k, g, Q, z_obs, std_obs, mode_centers, mode_radius, amp_est, Σ_obs, additive, nothing, nothing, nothing, nothing)

const HALF_WIDTH = 5.

bounds(p::LatentProblem) = (fill(-HALF_WIDTH, p.d), fill(HALF_WIDTH, p.d))
prior(p::LatentProblem) = something(p.x_prior, product_distribution(truncated.(Ref(Normal(0., HALF_WIDTH / 3)), bounds(p)...)))
likelihood_model(p::LatentProblem) = something(p.likelihood, isnothing(p.Σ_obs) ? NormalLikelihood(; p.z_obs, p.std_obs) : MvNormalLikelihood(; p.z_obs, p.Σ_obs))
obs_distribution(p::LatentProblem) = MvNormal(p.z_obs, isnothing(p.Σ_obs) ? Diagonal(abs2.(p.std_obs)) : p.Σ_obs)
y_size(p::LatentProblem) = length(p.z_obs)

informed(p::LatentProblem, X::AbstractMatrix) = (p.Q * X)[1:p.k, :]
simulator(p::LatentProblem) = x -> p.g((p.Q * x)[1:p.k])

function true_loglike(p::LatentProblem, X::AbstractMatrix)
    Y = reduce(hcat, p.g.(eachcol(informed(p, X))))
    return isnothing(p.loglike) ? logpdf(obs_distribution(p), Y) : p.loglike.(eachcol(Y))
end

"Initial design of run `seed` in the coordinates `[-5, 5]^d`, or `nothing` if the problem has no stored starting points."
function start_points(p::LatentProblem, seed::Int; dir="starts")
    file = joinpath(dir, p.name, "start_$seed.jld2")
    (isnothing(p.box) || !isfile(file)) && return nothing
    return to_unit(load(file, "X"), p.box...)
end

"Rescales the box `[lb, ub]` to `[-5, 5]^d`."
to_unit(X, lb, ub) = 2HALF_WIDTH .* (X .- lb) ./ (ub .- lb) .- HALF_WIDTH
from_unit(V, lb, ub) = lb .+ (ub .- lb) .* (V .+ HALF_WIDTH) ./ 2HALF_WIDTH

function gp_priors(p::LatentProblem)
    lb, ub = bounds(p)
    λ_min, λ_max = (ub .- lb) ./ 20, (ub .- lb) ./ 2
    amp = p.amp_est .* ones(y_size(p))
    return (;
        lengthscale_priors = fill(product_distribution(calc_inverse_gamma.(λ_min, λ_max)), y_size(p)),
        amplitude_priors = calc_inverse_gamma.(amp ./ 5, amp .* 2),
        noise_std_priors = fill(Dirac(0.), y_size(p)),
    )
end

function random_rotation(d::Int, seed::Int)
    Q = Matrix(qr(randn(MersenneTwister(seed), d, d)).Q)
    return Q
end
