# usage: julia --project=. scripts/run.jl <problem> <d> <seed> [budget]
include("../src/include_code.jl")

name, d, seed = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3])
budget = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 200

p = PROBLEMS[name](d)
res = run_experiment(p, Baseline(); seed, budget)
ref, ess = cached_reference(p)
@info "Reference ESS: $ess"
@info "Result" evaluate(p, res.samples, ref)...
