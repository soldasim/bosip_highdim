## Copies the TV curves of the BOSIP paper (runs 1-3, GP and WarpedGP) from the `bosip_benchmarks` repository.
## Run once; the plots then read `paper_tv/paper_tv.jld2`, so no dependency on that repository remains.

using JLD2

const PAPER_TV_FILE = "paper_tv/paper_tv.jld2"

const PAPER_RUNS = [
    ("rosenbrock5_cross", "data-opt-functions/RosenbrockProblem5_cross", "maxvar", "data-warpedgp2/RosenbrockProblem5_cross"),
    ("styblinski_tang5_cross", "data-opt-functions/StyblinskiTangProblem5_cross", "maxvar", "data-warpedgp2/StyblinskiTangProblem5_cross"),
    ("michalewicz5_cross", "data-opt-functions/MichalewiczProblem5_cross", "maxvar", "data-warpedgp2/MichalewiczProblem5_cross"),
    ("sphere5_cross", "data-opt-functions/SphereProblem5_cross", "maxvar", "data-warpedgp2/SphereProblem5_cross"),
    ("duffing5", "data-bosip-norm/DuffingProblem5", "standard", "data-warpedgp2/DuffingProblem5"),
    ("diffusion5d", "data-bosip-norm/DiffusionProblem5D", "standard", "data-warpedgp2/DiffusionProblem5D"),
]

"Number of scores kept per run: simulation counts `3:203`."
const PAPER_SCORES = 201

paper_scores(dir, run_name, seed) = jldopen(f -> f["score"][1:PAPER_SCORES], joinpath(dir, "$(run_name)_$(seed)_TVmetric.jld2"))

function copy_paper_tv(bench; seeds=1:3)
    paper = Dict(
        name => Dict(
            "gp" => [paper_scores(joinpath(bench, gp_dir), gp_run, s) for s in seeds],
            "warped_gp" => [paper_scores(joinpath(bench, wgp_dir), "warpedgp-yja-maxvar", s) for s in seeds],
        ) for (name, gp_dir, gp_run, wgp_dir) in PAPER_RUNS
    )
    mkpath(dirname(PAPER_TV_FILE))
    jldsave(PAPER_TV_FILE; paper, n = collect(3:2+PAPER_SCORES))
    return paper
end
