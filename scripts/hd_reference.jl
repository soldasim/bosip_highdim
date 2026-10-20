include("../src/include_code.jl")

"""
Computes the reference posterior of a 5D problem by SMC and caches it as `data/reference_<name>.jld2`.
A second independent run checks the reference: its sliced Wasserstein distance to the first one is the noise floor of the metric.
"""
function hd_reference(name; particles=4000, kwargs...)
    p = PROBLEMS[name](5)
    file = joinpath("data", "reference_$(p.name).jld2")
    isfile(file) && return @info "Reference exists" file
    t = @elapsed samples, info = smc_reference(p; particles, kwargs...)
    @info "Reference" name t info
    t_check = @elapsed check, info_check = smc_reference(p; particles, rng = MersenneTwister(2), kwargs...)
    halves = (samples[:, 1:end÷2], samples[:, end÷2+1:end])
    noise = (; run_vs_run = sliced_wasserstein(p, samples, check), half_vs_half = sliced_wasserstein(p, halves...))
    @info "Reference check" name t_check info_check noise
    mkpath("data")
    jldsave(file; samples, ess = info)
    jldsave(joinpath("data", "reference_check_$(p.name).jld2"); check, info, info_check, noise, t, t_check)
end
