## Wraps arbitrary BIPBM problems (Gaussian likelihoods) as `LatentProblem`s on `[-5, 5]^d` with all coordinates informed.

bosip_likelihood(l::BIPBM.AbstractLikelihood) =
    throw(ArgumentError("The likelihood `$(nameof(typeof(l)))` is not supported, only `GaussianLikelihood` and `MvGaussianLikelihood` are."))

"Prior of `bip` in the coordinates `[-5, 5]^d`; supports priors that are already there or uniform on the bounds."
function unit_prior(bip::BIPBM.AbstractBIPProblem)
    lb, ub = BIPBM.bounds(bip)
    prior = BIPBM.prior(bip)
    all(lb .== -HALF_WIDTH) && all(ub .== HALF_WIDTH) && return prior
    uniform = prior isa Distributions.Product && all(zip(prior.v, lb, ub)) do (u, l, h)
        u isa Uniform && u.a == l && u.b == h
    end
    uniform || throw(ArgumentError("Only problems with a uniform prior on the bounds (or bounds `[-5, 5]^d`) are supported."))
    return product_distribution(fill(Uniform(-HALF_WIDTH, HALF_WIDTH), length(lb)))
end

"Standard deviation of the simulator outputs at `count` prior samples (failed simulations are dropped); not counted in any budget."
function estimate_amplitude(f, prior, count, rng)
    Y = reduce(hcat, f.(eachcol(rand(rng, prior, count))))
    Y = Y[:, all.(isfinite, eachcol(Y))]
    return max.(vec(std(Y; dims=2)), 1e-3)
end

"""
    bipbm_problem(bip; name, amp_est, amp_samples, rng)

`LatentProblem` of the BIPBM problem `bip` with `k = d` (all coordinates informed, no modes known).
The coordinates are rescaled to `[-5, 5]^d` unless they already are.

## Keywords
- `amp_est`: GP amplitude estimate per output; by default the standard deviation of the outputs at `amp_samples` prior samples.
"""
function bipbm_problem(bip::BIPBM.AbstractBIPProblem; name=BIPBM.name(bip), amp_est=nothing, amp_samples=50, rng=Random.default_rng())
    lb, ub = BIPBM.bounds(bip)
    d = length(lb)
    unit = all(lb .== -HALF_WIDTH) && all(ub .== HALF_WIDTH)
    f = BIPBM.simulator(bip)
    sim = unit ? f : x -> f(from_unit(x, lb, ub))
    x_prior = unit_prior(bip)
    l = BIPBM.likelihood(bip)
    like = bosip_likelihood(l)
    amp = something(amp_est, estimate_amplitude(sim, x_prior, amp_samples, rng))
    return LatentProblem(
        name, d, d, sim, Matrix{Float64}(I, d, d),
        BIPBM.observation(l), noise_std(l), zeros(d, 0), 1., amp, noise_cov(l), false,
        x_prior, like, y -> BIPBM.loglike(l, y), unit ? nothing : (lb, ub),
    )
end

"`problem` holds the keywords passed to the BIPBM constructor (e.g. `(; d = 20)`)."
bipbm_problem(id::AbstractString; problem=(;), kwargs...) = bipbm_problem(BIPBM.problem(id; problem...); kwargs...)
