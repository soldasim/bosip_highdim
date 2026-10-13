include("pilot.jl")
p = get_problem("ab_emb", 10; rotate=false)
for m in (Capped(Baseline(), 5.), Capped(SaasMAP(), 5.), Capped(DimScaledMAP(), 5.))
    t = @elapsed res = run_experiment(p, m; budget = 30, n_init = 20, save = false)
    λ = res.bosip.problem.params.params.λ[:, 1]
    println(method_name(m), "  ", round(t; digits = 1), " s  max λ = ", round(maximum(λ); digits = 2), "  (cap = ", 5 * 10, ")  λ = ", round.(λ; digits = 1))
end
