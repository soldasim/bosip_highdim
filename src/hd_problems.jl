## The six 5D problems of the BOSIP paper (copied from the `bosip_benchmarks` repository).
## Coordinates are rescaled from the original parameter box to `[-5, 5]^5`.

"""
    rescaled_problem(name, f, box, marginals, z_obs, std_obs, amp_est; likelihood, loglike)

`LatentProblem` with all `d` coordinates informed, whose simulator `f` takes the original parameters in `box = (lb, ub)`.
`marginals` has an entry `(μ, σ)` (a normal truncated to the box) or `nothing` (uniform) per original parameter.
"""
function rescaled_problem(name, f, box, marginals, z_obs, std_obs, amp_est; likelihood, loglike)
    lb, ub = box
    d = length(lb)
    prior = product_distribution(map(lb, ub, marginals) do l, u, m
        isnothing(m) ? Uniform(-HALF_WIDTH, HALF_WIDTH) :
            truncated(Normal(to_unit(m[1], l, u), 2HALF_WIDTH * m[2] / (u - l)), -HALF_WIDTH, HALF_WIDTH)
    end)
    g = v -> f(from_unit(v, lb, ub))
    return LatentProblem(name, d, d, g, Matrix{Float64}(I, d, d), z_obs, std_obs, zeros(d, 0), 1., amp_est, nothing, false,
        prior, likelihood, loglike, (lb, ub))
end

normal_loglike(z_obs, std_obs) = (d = MvNormal(z_obs, Diagonal(abs2.(std_obs))); y -> logpdf(d, y))
lognormal_loglike(log_z_obs, CV) = (σ = sqrt.(log.(1 .+ abs2.(CV))); y -> sum(logpdf.(LogNormal.(y .- σ .^ 2 ./ 2, σ), exp.(log_z_obs))))

## Optimisation functions with the cross-polytope observation

rosenbrock(x) = sum(@. 100 * (x[2:end] - x[1:end-1]^2)^2 + (1 - x[1:end-1])^2)
styblinski_tang(x) = sum(@. (x^4 - 16x^2 + 5x) / 2)
michalewicz(x; m=10) = -sum(sin(xi) * sin(i * xi^2 / π)^(2m) for (i, xi) in enumerate(x))
sphere(x) = sum(abs2, x)

"""
    cross_polytope_problem(name, f, box, x_true, std_obs, amp_est)

The `2d` outputs are `f(x ± r eᵢ)` with `r` 2.5 % of the smallest box width; the observation is generated at `x_true`.
"""
function cross_polytope_problem(name, f, box, x_true, std_obs, amp_est)
    lb, ub = box
    d = length(lb)
    r = 0.025 * minimum(ub .- lb)
    offsets = [s * r * (1:d .== i) for i in 1:d for s in (1, -1)]
    sim = x -> [f(x .+ δ) for δ in offsets]
    z_obs = sim(x_true)
    std = fill(std_obs, 2d)
    return rescaled_problem(name, sim, box, fill(nothing, d), z_obs, std, fill(amp_est, 2d);
        likelihood = NormalLikelihood(; z_obs, std_obs = std), loglike = normal_loglike(z_obs, std))
end

## Duffing oscillator

module Duffing5
using DifferentialEquations

const x_ref = [0.15, -1.0, 0.5, 0.65, 2π]
const n_obs = 5
const std_obs = fill(0.1, n_obs)

function ode!(du, u, p, t)
    δ, α, β, γ, ω = p
    du[1] = u[2]
    du[2] = -δ * u[2] - α * u[1] - β * u[1]^3 + γ * cos(ω * t)
end

function simulator(x)
    sol = solve(ODEProblem(ode!, [0., 0.], (0., 10.), x), Tsit5(); saveat = 0.05)
    positions = [u[1] for u in sol.u[2:end]]
    return positions[(length(positions) ÷ n_obs):(length(positions) ÷ n_obs):end][1:n_obs]
end

const z_obs = simulator(x_ref)
const box = ([0.05, -2.0, 0.05, 0.10, π], [1.00, 2.0, 2.00, 1.50, 4π])
const marginals = [(0.10, 0.50), (0.00, 1.00), (0.00, 1.00), (0.65, 0.50), (2π, π)]
end

duffing5_problem() = rescaled_problem("duffing5", Duffing5.simulator, Duffing5.box, Duffing5.marginals,
    Duffing5.z_obs, Duffing5.std_obs, fill(2., Duffing5.n_obs);
    likelihood = NormalLikelihood(; z_obs = Duffing5.z_obs, std_obs = Duffing5.std_obs),
    loglike = normal_loglike(Duffing5.z_obs, Duffing5.std_obs))

## Advection-diffusion

module Diffusion5D
using DifferentialEquations

const x_grid = collect(range(-5., 5.; length = 51))
const y_grid = x_grid
const nx = length(x_grid)
const ny = nx
const dx = (x_grid[end] - x_grid[begin]) / (nx - 1)
const dy = dx
const t_span = (-10., 0.)
const sim_step = 0.1
const obs_points = [(-2., -2.), (2., -2.), (0., 1.4)]
const model_interval = 2.
const n_intervals = Int((t_span[2] - t_span[1]) / model_interval)
const n_obs = length(obs_points) * n_intervals
const CV = fill(0.1, n_obs)

const A_adv = 10.
const Δt_source = 0.5
const σ_source = 0.5
const A_source = 100.
const INIT_CONCENTRATION = 1e-2
const MIN_MEASURED_VALUE_PER_TIME_UNIT = 1e-2 / 10.
const wind_scale = 2.

const unit_v_x_field = [sin(wind_scale * y_grid[j] / 5.) * cos(wind_scale * x_grid[i] / 5.) for j in 1:ny, i in 1:nx]
const unit_v_y_field = [-cos(wind_scale * y_grid[j] / 5.) * sin(wind_scale * x_grid[i] / 5.) for j in 1:ny, i in 1:nx]

const x_ref = [1.5, 1.5, -4.0, 0.05, 0.8]

function pde!(du, u, p, t)
    x_s, y_s, t_s, D_, v_max_ = p
    fill!(du, 0.)
    source_active = (t_s <= t < t_s + Δt_source)

    for j in 1:ny, i in 1:nx
        idx = (j - 1) * nx + i

        d2u_dx2 = if i == 1
            (u[idx+1] - 2 * u[idx] + u[idx]) / dx^2
        elseif i == nx
            (u[idx] - 2 * u[idx] + u[idx-1]) / dx^2
        else
            (u[idx+1] - 2 * u[idx] + u[idx-1]) / dx^2
        end

        d2u_dy2 = if j == 1
            (u[idx+nx] - 2 * u[idx] + u[idx]) / dy^2
        elseif j == ny
            (u[idx] - 2 * u[idx] + u[idx-nx]) / dy^2
        else
            (u[idx+nx] - 2 * u[idx] + u[idx-nx]) / dy^2
        end

        v_x = v_max_ * unit_v_x_field[j, i]
        v_y = v_max_ * unit_v_y_field[j, i]

        du_dx = if i == 1
            v_x >= 0 ? zero(eltype(u)) : (u[idx+1] - u[idx]) / dx
        elseif i == nx
            v_x >= 0 ? (u[idx] - u[idx-1]) / dx : zero(eltype(u))
        else
            v_x >= 0 ? (u[idx] - u[idx-1]) / dx : (u[idx+1] - u[idx]) / dx
        end

        du_dy = if j == 1
            v_y >= 0 ? zero(eltype(u)) : (u[idx+nx] - u[idx]) / dy
        elseif j == ny
            v_y >= 0 ? (u[idx] - u[idx-nx]) / dy : zero(eltype(u))
        else
            v_y >= 0 ? (u[idx] - u[idx-nx]) / dy : (u[idx+nx] - u[idx]) / dy
        end

        source = if source_active
            A_source * exp(-((x_grid[i] - x_s)^2 + (y_grid[j] - y_s)^2) / (2 * σ_source^2))
        else
            zero(eltype(u))
        end

        du[idx] = D_ * (d2u_dx2 + d2u_dy2) - A_adv * (v_x * du_dx + v_y * du_dy) + source
    end
end

function solution(x)
    u0 = INIT_CONCENTRATION * ones(eltype(x), nx * ny)
    prob = ODEProblem(pde!, u0, (max(t_span[1], x[3]), t_span[2]), x)
    return solve(prob, Tsit5(); saveat = sim_step)
end

function measurements(sol)
    obs_indices = [(findfirst(==(x), x_grid), findfirst(==(y), y_grid)) for (x, y) in obs_points]
    y = zeros(eltype(sol.u[1]), n_obs)

    for interval_idx in 1:n_intervals
        t_start = t_span[1] + (interval_idx - 1) * model_interval
        t_end = t_start + model_interval

        (t_end <= sol.t[begin]) && continue
        t_start = max(t_start, sol.t[begin])

        start_idx = findfirst(t -> t >= t_start, sol.t)
        end_idx = findfirst(t -> t >= t_end, sol.t)
        end_idx = something(end_idx, length(sol.t) + 1) - 1

        for (obs_idx, (i, j)) in enumerate(obs_indices)
            grid_idx = (j - 1) * nx + i
            integrated = 0.

            if start_idx > 1
                c1, c2 = sol.u[start_idx-1][grid_idx], sol.u[start_idx][grid_idx]
                integrated += 0.5 * (c1 + c2) * (sol.t[start_idx] - t_start)
            end
            for t_idx in start_idx:end_idx
                dt = sol.t[t_idx+1] - sol.t[t_idx]
                c1, c2 = sol.u[t_idx][grid_idx], sol.u[t_idx+1][grid_idx]
                integrated += 0.5 * (c1 + c2) * dt
            end
            if end_idx < length(sol.t)
                c1, c2 = sol.u[end_idx][grid_idx], sol.u[end_idx+1][grid_idx]
                integrated -= 0.5 * (c1 + c2) * (sol.t[end_idx+1] - t_end)
            end

            y[(obs_idx-1)*n_intervals+interval_idx] = integrated
        end
    end

    return max.(MIN_MEASURED_VALUE_PER_TIME_UNIT * model_interval, y)
end

simulator(x) = log.(measurements(solution(x)))

const log_z_obs = simulator(x_ref)
const box = ([-5., -5., -10., 0.01, 0.10], [5., 5., -0.5, 0.20, 2.00])
const marginals = [nothing, nothing, (-2., 5.), (0.05, 0.05), (0.80, 0.50)]
end

diffusion5d_problem() = rescaled_problem("diffusion5d", Diffusion5D.simulator, Diffusion5D.box, Diffusion5D.marginals,
    Diffusion5D.log_z_obs, Diffusion5D.CV, fill(log(Diffusion5D.A_source * Diffusion5D.model_interval), Diffusion5D.n_obs);
    likelihood = LogNormalLikelihood(; log_z_obs = Diffusion5D.log_z_obs, CV = Diffusion5D.CV),
    loglike = lognormal_loglike(Diffusion5D.log_z_obs, Diffusion5D.CV))

## Registry

const HD_PROBLEMS = Dict(
    "rosenbrock5_cross" => () -> cross_polytope_problem("rosenbrock5_cross", rosenbrock, (fill(-2., 5), fill(2., 5)), zeros(5), 0.5, 100.),
    "styblinski_tang5_cross" => () -> cross_polytope_problem("styblinski_tang5_cross", styblinski_tang, (fill(-5., 5), fill(5., 5)),
        [-3.248518, 3.084936, -3.248518, 3.084936, -3.248518], 8., 50.),
    "michalewicz5_cross" => () -> cross_polytope_problem("michalewicz5_cross", michalewicz, (fill(0., 5), fill(float(π), 5)),
        [2.097956, 1.632165, 1.28, 1.11, 0.99], 0.3, 1.),
    "sphere5_cross" => () -> cross_polytope_problem("sphere5_cross", sphere, (fill(-5.12, 5), fill(5.12, 5)), [1., -1., 1.2, -0.8, 1.], 0.5, 52.),
    "duffing5" => duffing5_problem,
    "diffusion5d" => diffusion5d_problem,
)

for (name, make) in HD_PROBLEMS
    PROBLEMS[name] = (args...; kwargs...) -> make()
end
