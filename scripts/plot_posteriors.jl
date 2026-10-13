include("plot_curves.jl")
include("../src/include_code.jl")

const POSTERIOR_PROBLEMS = [("ab_emb", "ab"), ("simple_emb", "simple"), ("banana_emb", "banana"), ("bimodal_emb", "bimodal"), ("wavy_ridge", "ridge"), ("wavy_multimodal", "multimodal")]
const POSTERIOR_ROWS = [("aligned", false, first.(POSTERIOR_PROBLEMS)), ("rotated", true, first.(POSTERIOR_PROBLEMS)), ("additive, all 10 relevant (first 2 coords)", false, ["ab_sum", "simple_sum", "banana_sum", "bimodal_sum"])]

"Coordinates shown for a problem: informed coords (first two, or one for `k = 1`)."
shown(p::LatentProblem, X) = informed(p, X)[1:min(p.k, 2), :]

"Reference (grey) vs. final samples of `method` (coloured) in the informed coordinates."
function plot_posteriors(; method="dsp_map", seed=1, file="plots/posterior_$(replace(method_family(method), "_cap" => ""))", dir="data")
    fam = method_family(method)
    fig = Figure(; size = (1980, 1000), backgroundcolor = SURFACE, figure_padding = (16, 16, 12, 12))
    for (i, (label, rotate, names)) in enumerate(POSTERIOR_ROWS), (j, name) in enumerate(names)
        p = PROBLEMS[name](10; rotate)
        ax = Axis(fig[i + 1, j]; title = "$(replace(name, r"^wavy_|_emb$|_sum$" => "")) – $label", xlabel = "x₁", ylabel = j == 1 ? "x₂" : "")
        style_axis!(ax)
        ref, _ = cached_reference(p; dir)
        file_ = joinpath(dir, p.name, "$(method)_$(seed)_samples.jld2")
        X = isfile(file_) ? load(file_, "samples") : nothing
        clamped = isfile(file_) && get(load(file_), "clamped", false)
        R = shown(p, ref)
        if size(R, 1) == 1
            hist!(ax, vec(R); bins = 40, normalization = :pdf, color = (INK2, 0.35))
            isnothing(X) || stephist!(ax, vec(shown(p, X)); bins = 40, normalization = :pdf, color = COLORS[fam], linewidth = 2)
            ax.ylabel = "density"
        else
            scatter!(ax, R[1, :], R[2, :]; color = (INK2, 0.35), markersize = 4)
            isnothing(X) || scatter!(ax, shown(p, X)[1, :], shown(p, X)[2, :]; color = (COLORS[fam], 0.5), markersize = 4)
            xlims!(ax, -HALF_WIDTH, HALF_WIDTH); ylims!(ax, -HALF_WIDTH, HALF_WIDTH)
        end
        clamped && text!(ax, 0.02, 0.98; text = "λ clamped at 5×width for sampling", space = :relative, align = (:left, :top), color = :red, fontsize = 11)
        isnothing(X) && text!(ax, 0.5, 0.5; text = "no samples", space = :relative, align = (:center, :center), color = INK2)
    end
    Label(fig[0, 1:6], "Final posterior approximation ($(LABELS[fam]), seed $seed, 200 simulations) vs. true posterior"; color = INK, fontsize = 15, halign = :left, font = :bold)
    elems = [MarkerElement(color = (INK2, 0.6), marker = :circle), MarkerElement(color = COLORS[fam], marker = :circle)]
    Legend(fig[1, 1:6], elems, ["True posterior (reference samples)", LABELS[fam]]; orientation = :horizontal, framevisible = false, labelcolor = INK2, halign = :left)
    mkpath(dirname(file))
    save(file * ".png", fig; px_per_unit = 2)
    save(file * ".pdf", fig)
    return fig
end

const POSTERIOR_METHODS = ["baseline", "saas_map_a0.1", "dsp_map", "baseline_cap5.0", "saas_map_a0.1_cap5.0", "dsp_map_cap5.0", "latent_k3", "latent_ktrue", "oracle_gp"]
"One posterior figure per BOSIP variant."
plot_all_posteriors(; kwargs...) = foreach(m -> plot_posteriors(; method = m, kwargs...), POSTERIOR_METHODS)

const TRUNCATED_METHODS = ["baseline_cap5.0", "saas_map_a0.1_cap5.0", "dsp_map_cap5.0", "latent_k3", "latent_ktrue", "oracle_gp"]
"Posterior figures for the lengthscale-truncated variants only."
plot_truncated_posteriors(; kwargs...) = foreach(m -> plot_posteriors(; method = m, kwargs...), TRUNCATED_METHODS)
