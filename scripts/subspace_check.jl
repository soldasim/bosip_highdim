## Compares the learned encoder subspaces of the hidden-latent problems with the true informed subspace.

include("../src/include_code.jl")
include("pilot.jl")

problem_from_dir(dir) = (m = match(r"^(.*)_d(\d+)(_aligned)?$", dir); get_problem(m[1], parse(Int, m[2]); rotate = isnothing(m[3])))

"Squared-cosine overlap (see `subspace_overlap`) per output at iteration `it` of a stored encoder run."
function run_overlaps(p, model, file; it=nothing)
    params = jldopen(f -> f["params"], file)
    ps = params[something(it, length(params))].params
    return [subspace_overlap(p, encoder(model, w)) for w in eachcol(ps.W)]
end

function subspace_check(; dir="data", methods=("linenc_k3", "linenc_ktrue", "linenc_vae_k3", "linenc_vae_ktrue"), seeds=1:3)
    for pd in sort(filter(d -> occursin(r"_d\d+(_aligned)?$", d) && !occursin("_sum_", d) && isdir(joinpath(dir, d)), readdir(dir)))
        p = problem_from_dir(pd)
        for m in methods
            k = endswith(m, "k3") ? 3 : p.k
            model = latent_model(LinearEncoder(; k, vae = occursin("vae", m) ? 0.5 : 0.), p)
            ov = [mean(run_overlaps(p, model, joinpath(dir, pd, "$(m)_$(s)_iters.jld2"))) for s in seeds if isfile(joinpath(dir, pd, "$(m)_$(s)_iters.jld2"))]
            isempty(ov) || println(rpad(pd, 26), rpad(m, 16), "k=$k true=$(p.k) chance=$(round(k / p.d; digits = 2))  overlap per seed: ", round.(ov; digits = 3))
        end
    end
end
