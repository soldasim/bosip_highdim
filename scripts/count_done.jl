using JLD2
n = sum(f -> isfile(f) ? length(load(f, "results")) : 0, "data/pilot_curves_latent_" .* ["ridge", "banana", "multi"] .* ".jld2")
println(n)
