include("../src/include_code.jl")
p = multimodal_problem(4)
res = run_experiment(p; budget = 8, n_init = 5, sample_count = 200, save = false)
ref, ess = cached_reference(p; dir = mktempdir(), chunk = 10^5, chunks = 2)
@info "ESS" ess
@info "metrics" evaluate(p, res.samples, ref)...
