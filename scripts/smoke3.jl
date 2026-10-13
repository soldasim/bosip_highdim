include("pilot.jl")
sp = "/private/tmp/claude-501/-Users-soldasim/d356b4a5-cf2a-49c9-bf97-0d13aadcdd8c/scratchpad"
for (prob, rot, m) in [("banana_sum", false, OracleGP()), ("banana_sum", false, LatentGP()), ("banana_sum", false, SaasMAP()), ("banana_emb", true, OracleGP()), ("simple_emb", true, LatentGP(; k=nothing))]
    pilot(; problems=[(prob, rot)], methods=[m], seeds=[99], budget=25, file="$sp/smoke3_$(prob)_$(method_name(m)).jld2")
end
