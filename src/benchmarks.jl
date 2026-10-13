function make_problem(name, g, k, d, z_obs, std_obs; centers=zeros(k, 0), radius=1., rotate=true, seed=1, amp_est=5., Σ_obs=nothing, additive=false)
    Q = rotate ? random_rotation(d, seed) : Matrix{Float64}(I, d, d)
    return LatentProblem("$(name)_d$(d)" * (rotate ? "" : "_aligned"), d, k, g, Q, z_obs, std_obs, centers, radius, amp_est, Σ_obs, additive)
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

"Slab: one informed direction."
ridge_problem(d; kwargs...) = make_problem("ridge", v -> [v[1]], 1, d, [1.5], [0.3]; kwargs...)

"Curved ridge in two informed directions."
banana_problem(d; kwargs...) = make_problem("banana", v -> [v[2] + 0.2 * v[1]^2], 2, d, [1.], [0.3]; kwargs...)

"Four separated modes in two informed directions."
multimodal_problem(d; kwargs...) = make_problem("multimodal", v -> v .^ 2, 2, d, [4., 4.], [0.5, 0.5];
    centers = [2. 2. -2. -2.; 2. -2. 2. -2.], kwargs...)

"Non-monotone slab (several slabs)."
wavy_ridge_problem(d; kwargs...) = make_problem("wavy_ridge", v -> [v[1] + sin(2 * v[1])], 1, d, [1.5], [0.3]; kwargs...)

"Sinusoidal banana."
wavy_banana_problem(d; kwargs...) = make_problem("wavy_banana", v -> [v[2] + 2 * sin(v[1] / 2)], 2, d, [1.], [0.3]; kwargs...)

"Four modes, non-polynomial simulator."
function wavy_multimodal_problem(d; kwargs...)
    h(v) = v^2 + 2 * sin(2 * v)
    z = 4.
    rs = scalar_roots(h, z)
    centers = reduce(hcat, [[a, b] for a in rs for b in rs])
    return make_problem("wavy_multimodal", v -> h.(v), 2, d, [z, z], [0.5, 0.5]; centers, radius=0.5, kwargs...)
end

# Problems from the BOSIP paper: original simulator and likelihood, with the original box
# mapped onto `[-HALF_WIDTH, HALF_WIDTH]` and a truncated Normal prior in the mapped coordinates.
struct PaperProblem{F}
    name::String
    f::F
    lb::Vector{Float64}
    ub::Vector{Float64}
    z_obs::Vector{Float64}
    Σ_obs::Matrix{Float64}
end

corr(ρ) = [1. ρ; ρ 1.]
const PAPER_PROBLEMS = Dict(
    "ab" => PaperProblem("ab", θ -> [θ[1] * θ[2]], [-5., -5.], [5., 5.], [1.], fill(0.2^2, 1, 1)),
    "simple" => PaperProblem("simple", θ -> θ, [-16., -16.], [16., 16.], [0., 0.], corr(0.25)),
    "banana" => PaperProblem("banana", θ -> [θ[1], θ[2] + θ[1]^2 + 1], [-6., -20.], [6., 2.], [0., 0.], corr(0.9)),
    "bimodal" => PaperProblem("bimodal", θ -> [θ[1], θ[2]^2 - 2], [-6., -6.], [6., 6.], [0., 0.], corr(0.5)),
)

to_params(pp::PaperProblem, u) = (pp.lb .+ pp.ub) ./ 2 .+ (pp.ub .- pp.lb) ./ 2 .* u ./ HALF_WIDTH

"Paper problem on the first two coordinates; the other `d - 2` are uninformed."
embedded_problem(pp::PaperProblem, d; kwargs...) =
    make_problem("$(pp.name)_emb", v -> pp.f(to_params(pp, v)), 2, d, pp.z_obs, sqrt.(diag(pp.Σ_obs)); Σ_obs = pp.Σ_obs, amp_est = 20., kwargs...)

"Mean of the paper problem over the `d / 2` consecutive coordinate pairs, as in the 6D extension of the papers."
function summed_problem(pp::PaperProblem, d; rotate=false, kwargs...)
    g(v) = mean(pp.f(to_params(pp, v[i:i+1])) for i in 1:2:d)
    return make_problem("$(pp.name)_sum", g, d, d, pp.z_obs, sqrt.(diag(pp.Σ_obs)); Σ_obs = pp.Σ_obs, amp_est = 20., additive = true, rotate = false, kwargs...)
end

const PROBLEMS = Dict(
    "ridge" => ridge_problem,
    "banana" => banana_problem,
    "multimodal" => multimodal_problem,
    "wavy_ridge" => wavy_ridge_problem,
    "wavy_banana" => wavy_banana_problem,
    "wavy_multimodal" => wavy_multimodal_problem,
)
for (n, pp) in PAPER_PROBLEMS
    PROBLEMS["$(n)_emb"] = (d; kwargs...) -> embedded_problem(pp, d; kwargs...)
    PROBLEMS["$(n)_sum"] = (d; kwargs...) -> summed_problem(pp, d; kwargs...)
end
