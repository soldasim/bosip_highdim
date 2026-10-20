include("../src/include_code.jl")

for (name, method) in [("rosenbrock5_cross", LinearEncoder(; k=3)), ("diffusion5d", LinearEncoder(; k=3)), ("duffing5", Capped(DimScaledMAP(), 5.))]
    p = PROBLEMS[name](5)
    t = @elapsed res = run_experiment(p, method; seed = 1, budget = 8, ref = nothing, save = false, sample_count = 200)
    @info "Smoke run" name method = method_name(method) time = t n = size(res.bosip.problem.data.X, 2) samples = size(res.samples)
end
