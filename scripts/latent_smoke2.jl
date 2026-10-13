include("pilot.jl")
for m in (LatentGP(), OracleGP())
    r = pilot(; problems=[("wavy_banana", true)], methods=[m], seeds=[99], budget=30, score_every=5, file="/private/tmp/claude-501/-Users-soldasim/d356b4a5-cf2a-49c9-bf97-0d13aadcdd8c/scratchpad/smoke_$(method_name(m)).jld2")
end
