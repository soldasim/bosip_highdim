include("../src/include_code.jl")

for (name, d, method) in [("ab_emb", 10, LinearEncoder(; k=3)), ("rosenbrock5_cross", 5, LinearEncoder(; k=3)), ("rosenbrock5_cross", 5, Capped(DimScaledMAP(), 5.)), ("duffing5", 5, LinearEncoder(; k=3)), ("diffusion5d", 5, LinearEncoder(; k=3))]
    p = PROBLEMS[name](d; rotate = true)
    try
        t = @elapsed res = run_experiment(p, method; seed = 1, budget = p.d == 5 ? 8 : 22, ref = nothing, save = false, sample_count = 200)
        @info "OK" name method_name(method) t n = size(res.bosip.problem.data.X, 2)
    catch e
        @info "FAILED" name method_name(method)
        println(first(sprint(showerror, e, catch_backtrace()), 1500))
    end
end
