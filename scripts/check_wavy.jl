include("../src/include_code.jl")
function check_wavy(d=10)
    p = wavy_multimodal_problem(d)
    @info "centers" p.mode_centers
    for name in ("wavy_ridge", "wavy_banana", "wavy_multimodal")
        q = PROBLEMS[name](d)
        ref, ess = cached_reference(q)
        V = informed(q, ref)
        @info name ess mean_v=round.(mean(V; dims=2)[:]; digits=2) std_v=round.(std(V; dims=2)[:]; digits=2)
        if size(q.mode_centers, 2) > 0
            @info "mode masses" [mean(sqrt.(sum(abs2, V .- c; dims=1)) .< q.mode_radius) for c in eachcol(q.mode_centers)]
        end
    end
end
