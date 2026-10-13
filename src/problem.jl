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
    amp_est::Float64
    Σ_obs::Union{Nothing, Matrix{Float64}}
    additive::Bool
end

const HALF_WIDTH = 5.

bounds(p::LatentProblem) = (fill(-HALF_WIDTH, p.d), fill(HALF_WIDTH, p.d))
prior(p::LatentProblem) = product_distribution(truncated.(Ref(Normal(0., HALF_WIDTH / 3)), bounds(p)...))
likelihood_model(p::LatentProblem) = isnothing(p.Σ_obs) ? NormalLikelihood(; p.z_obs, p.std_obs) : MvNormalLikelihood(; p.z_obs, p.Σ_obs)
obs_distribution(p::LatentProblem) = MvNormal(p.z_obs, isnothing(p.Σ_obs) ? Diagonal(abs2.(p.std_obs)) : p.Σ_obs)
y_size(p::LatentProblem) = length(p.z_obs)

informed(p::LatentProblem, X::AbstractMatrix) = (p.Q * X)[1:p.k, :]
simulator(p::LatentProblem) = x -> p.g((p.Q * x)[1:p.k])

function true_loglike(p::LatentProblem, X::AbstractMatrix)
    Y = reduce(hcat, p.g.(eachcol(informed(p, X))))
    return logpdf(obs_distribution(p), Y)
end

function gp_priors(p::LatentProblem)
    lb, ub = bounds(p)
    λ_min, λ_max = (ub .- lb) ./ 20, (ub .- lb) ./ 2
    return (;
        lengthscale_priors = fill(product_distribution(calc_inverse_gamma.(λ_min, λ_max)), y_size(p)),
        amplitude_priors = fill(calc_inverse_gamma(p.amp_est / 5, p.amp_est * 2), y_size(p)),
        noise_std_priors = fill(Dirac(0.), y_size(p)),
    )
end

function random_rotation(d::Int, seed::Int)
    Q = Matrix(qr(randn(MersenneTwister(seed), d, d)).Q)
    return Q
end
