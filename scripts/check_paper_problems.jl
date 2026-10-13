include("../src/include_code.jl")
for n in ["ab", "simple", "banana", "bimodal"], suffix in ["emb", "sum"], rotate in (suffix == "emb" ? (false, true) : (false,))
    p = PROBLEMS["$(n)_$(suffix)"](10; rotate)
    ref, ess = cached_reference(p)
    @info p.name ess=round(ess) ref_mean_abs=round.(mean(abs, informed(p, ref); dims=2)[1:min(3, p.k)]; digits=2) simulator_ok=length(simulator(p)(zeros(10)))
end
