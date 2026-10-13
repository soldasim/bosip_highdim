using JLD2
fs = filter(f -> occursin(r"^pilot_curves_paper_pp-", f), readdir("data"))
println(sum(f -> length(load(joinpath("data", f), "results")), fs; init=0))
