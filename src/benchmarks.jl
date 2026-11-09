## The simple benchmarks are defined in BIPBM ("sparse-*" and "additive-*" problems).
## Here they are wrapped into `LatentProblem`s with the experiment metadata (modes, amplitude estimates).

bosip_likelihood(l::BIPBM.GaussianLikelihood) = NormalLikelihood(; z_obs = l.z_obs, std_obs = l.std)
bosip_likelihood(l::BIPBM.MvGaussianLikelihood) = MvNormalLikelihood(; z_obs = l.z_obs, Σ_obs = l.Σ)

noise_std(l::BIPBM.GaussianLikelihood) = l.std
noise_std(l::BIPBM.MvGaussianLikelihood) = sqrt.(diag(l.Σ))
noise_cov(::BIPBM.GaussianLikelihood) = nothing
noise_cov(l::BIPBM.MvGaussianLikelihood) = l.Σ

"""
    latent_problem(bip, name, P; mode_centers, mode_radius, amp_est, additive)

`LatentProblem` of the BIPBM problem `bip` with informed coordinates `P * x`.
"""
function latent_problem(bip::BIPBM.BIPProblem, name::String, P::Matrix{Float64}; mode_centers=zeros(size(P, 1), 0), mode_radius=1., amp_est=5., additive=false)
    l = BIPBM.likelihood(bip)
    return LatentProblem(
        name, BIPBM.x_dim(bip), size(P, 1), BIPBM.simulator(bip), P,
        BIPBM.observation(l), noise_std(l), mode_centers, mode_radius, amp_est, noise_cov(l), additive,
        BIPBM.prior(bip), bosip_likelihood(l), y -> BIPBM.loglike(l, y), nothing,
    )
end

"Roots of `h(v) = z` on `[-5, 5]` found by scanning for sign changes and bisecting."
function scalar_roots(h, z; lo=-5., hi=5., n=2001)
    vs = range(lo, hi; length=n)
    roots = Float64[]
    for (a, b) in zip(vs[1:end-1], vs[2:end])
        (h(a) - z) * (h(b) - z) > 0 && continue
        for _ in 1:60
            m = (a + b) / 2
            (h(a) - z) * (h(m) - z) > 0 ? (a = m) : (b = m)
        end
        push!(roots, (a + b) / 2)
    end
    return roots
end

## Modes of the true posterior in the informed coordinates (empty if there is no small set of separated modes).
sparse_modes(::Val, k, d) = (; mode_centers = zeros(k, 0), mode_radius = 1.)
sparse_modes(::Val{Symbol("sparse-multimodal")}, k, d) = (; mode_centers = [2. 2. -2. -2.; 2. -2. 2. -2.], mode_radius = 1.)
function sparse_modes(::Val{Symbol("sparse-wavy-multimodal")}, k, d)
    aligned = BIPBM.problem("sparse-wavy-multimodal"; d, rotate=false)
    rs = scalar_roots(v -> BIPBM.simulator(aligned)(vcat(v, zeros(d - 1)))[1], BIPBM.observation(BIPBM.likelihood(aligned))[1])
    return (; mode_centers = reduce(hcat, [[a, b] for a in rs for b in rs]), mode_radius = 0.5)
end

"""
    sparse_problem(id, name; d=10, rotate=true, seed=1)

`LatentProblem` named `name` of the BIPBM problem `id` (`"sparse-*"`), with `d` coordinates of which `k` are informed.
"""
function sparse_problem(id::String, name::String; d=10, rotate=true, seed=1)
    bip = BIPBM.problem(id; d, rotate, seed)
    P = BIPBM.informed_projection(id; d, rotate, seed)
    modes = sparse_modes(Val(Symbol(id)), size(P, 1), d)
    amp_est = startswith(id, "sparse-bosip-") ? 20. : 5.
    return latent_problem(bip, "$(name)_d$(d)" * (rotate ? "" : "_aligned"), P; modes..., amp_est)
end

"""
    additive_problem(id, name; d=10)

`LatentProblem` named `name` of the BIPBM problem `id` (`"additive-*"`); all `d` coordinates are informed.
"""
function additive_problem(id::String, name::String; d=10)
    bip = BIPBM.problem(id; d)
    return latent_problem(bip, "$(name)_d$(d)_aligned", BIPBM.informed_projection(id; d); amp_est = 20., additive = true)
end

const PAPER_NAMES = ("ab", "simple", "banana", "bimodal")

const PROBLEMS = Dict{String, Function}(
    "ridge" => (d; kwargs...) -> sparse_problem("sparse-ridge", "ridge"; d, kwargs...),
    "banana" => (d; kwargs...) -> sparse_problem("sparse-banana", "banana"; d, kwargs...),
    "multimodal" => (d; kwargs...) -> sparse_problem("sparse-multimodal", "multimodal"; d, kwargs...),
    "wavy_ridge" => (d; kwargs...) -> sparse_problem("sparse-wavy-ridge", "wavy_ridge"; d, kwargs...),
    "wavy_banana" => (d; kwargs...) -> sparse_problem("sparse-wavy-banana", "wavy_banana"; d, kwargs...),
    "wavy_multimodal" => (d; kwargs...) -> sparse_problem("sparse-wavy-multimodal", "wavy_multimodal"; d, kwargs...),
)
for n in PAPER_NAMES
    PROBLEMS["$(n)_emb"] = (d; kwargs...) -> sparse_problem("sparse-bosip-$n", "$(n)_emb"; d, kwargs...)
    PROBLEMS["$(n)_sum"] = (d; rotate=false, kwargs...) -> additive_problem("additive-bosip-$n", "$(n)_sum"; d)
end

ridge_problem(d; kwargs...) = PROBLEMS["ridge"](d; kwargs...)
banana_problem(d; kwargs...) = PROBLEMS["banana"](d; kwargs...)
multimodal_problem(d; kwargs...) = PROBLEMS["multimodal"](d; kwargs...)
wavy_ridge_problem(d; kwargs...) = PROBLEMS["wavy_ridge"](d; kwargs...)
wavy_banana_problem(d; kwargs...) = PROBLEMS["wavy_banana"](d; kwargs...)
wavy_multimodal_problem(d; kwargs...) = PROBLEMS["wavy_multimodal"](d; kwargs...)
