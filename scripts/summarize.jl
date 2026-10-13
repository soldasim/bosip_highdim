using JLD2, Statistics, Printf

files = length(ARGS) >= 1 ? ARGS : sort(filter(f -> occursin(r"^pilot_curves.*\.jld2$", basename(f)), readdir("data"; join=true)))
checkpoints = [40, 80, 120, 160, 200]
results = reduce(vcat, load.(files, "results"))
groups = Dict{Tuple{String,String},Vector{Any}}()
for r in results
    push!(get!(groups, (r.key[1], r.key[2]), Any[]), r)
end

sw_at(r, n) = (i = findfirst(s -> s.n == n, get(r, :scores, [])); isnothing(i) ? NaN : r.scores[i].metrics.sliced_wasserstein)
rec_at(r, n) = (i = findfirst(s -> s.n == n, get(r, :scores, [])); isnothing(i) ? NaN : r.scores[i].metrics.mode_recall)

println("Median sliced Wasserstein over seeds at n simulations (lower is better)")
@printf("%-28s %-22s %2s", "problem", "method", "k")
foreach(n -> @printf(" %7s", "n=$n"), checkpoints)
@printf(" %9s %12s %8s\n", "recall", "λ_oth/λ_inf", "overlap")
for k in sort(collect(keys(groups)))
    rs = groups[k]
    @printf("%-28s %-22s %2d", k..., length(rs))
    foreach(n -> @printf(" %7.3f", (v = filter(!isnan, sw_at.(rs, n)); isempty(v) ? NaN : median(v))), checkpoints)
    rec = mean(r.metrics.mode_recall for r in rs)
    ratio = median([r.ls.λ_other / r.ls.λ_informed for r in rs])
    overlap = median([get(r.ls, :overlap, NaN) for r in rs])
    @printf(" %9.2f %12.2f %8.2f\n", rec, ratio, overlap)
end
