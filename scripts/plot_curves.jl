using CairoMakie, JLD2, Statistics

# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode).
const SURFACE = "#fcfcfb"
const INK = "#0b0b0b"
const INK2 = "#52514e"
const GRID = "#e6e5e1"
const COLORS = Dict("baseline" => "#2a78d6", "saas_map" => "#eb6834", "dsp_map" => "#1baf7a", "saas_bi" => "#eda100",
    "latent_k3" => "#7b4fd1", "latent_ktrue" => "#c2368a", "oracle_gp" => "#0b0b0b")
const LABELS = Dict("baseline" => "Baseline GP", "saas_map" => "SAAS prior, MAP", "dsp_map" => "Dimension-scaled prior, MAP", "saas_bi" => "SAAS prior, NUTS",
    "latent_k3" => "Learned linear encoder, k=3", "latent_ktrue" => "Learned linear encoder, k=true", "oracle_gp" => "Oracle (true structure)")
for f in ("baseline", "saas_map", "dsp_map")
    COLORS[f * "_cap"] = COLORS[f]
    LABELS[f * "_cap"] = LABELS[f]
end
const RELEVANT_DIMS = Dict("wavy_ridge" => " (1 relevant)", "wavy_multimodal" => " (2 relevant)", [n * "_emb" => " (2 relevant)" for n in ("ab", "simple", "banana", "bimodal")]...)
const PROBLEMS_PLOT = ["wavy_ridge", "wavy_banana", "wavy_multimodal"]

method_family(m) = occursin("_cap", m) ? method_family(replace(m, r"_cap[\d.]+$" => "")) * "_cap" : startswith(m, "latent") ? m : m == "oracle_gp" ? m : startswith(m, "saas_map") ? "saas_map" : startswith(m, "dsp_map") ? "dsp_map" : startswith(m, "saas_bi") ? "saas_bi" : "baseline"
problem_base(name) = replace(name, r"_d\d+(_aligned)?$" => "")
is_aligned(name) = endswith(name, "_aligned")

load_results(dir="data") = reduce(vcat, [load(joinpath(dir, f), "results") for f in sort(readdir(dir)) if occursin(r"^pilot_curves.*\.jld2$", f)]; init=Any[])

"Median and range over seeds at each score checkpoint `n`."
function aggregate(rs, field)
    ns = sort(unique(s.n for r in rs for s in r.scores))
    vals = [[getproperty(s.metrics, field) for r in rs for s in r.scores if s.n == n] for n in ns]
    vals = [filter(!isnan, v) for v in vals]
    keep = .!isempty.(vals)
    return ns[keep], median.(vals[keep]), minimum.(vals[keep]), maximum.(vals[keep])
end

function style_axis!(ax)
    ax.backgroundcolor = SURFACE
    ax.xgridcolor = GRID; ax.ygridcolor = GRID
    ax.xgridwidth = 0.5; ax.ygridwidth = 0.5
    ax.topspinevisible = false; ax.rightspinevisible = false
    ax.leftspinecolor = INK2; ax.bottomspinecolor = INK2
    ax.xtickcolor = INK2; ax.ytickcolor = INK2
    ax.xticklabelcolor = INK2; ax.yticklabelcolor = INK2
    ax.titlecolor = INK; ax.xlabelcolor = INK2; ax.ylabelcolor = INK2
    ax.titlesize = 13; ax.titlealign = :left
end

"""
Small multiples of convergence curves. `rows` is a vector of `(label, aligned, problems)`;
a panel is drawn for every problem in every row.
"""
function plot_curves(results; field=:sliced_wasserstein, problems=PROBLEMS_PLOT, ylabel="sliced Wasserstein distance (lower is better)",
    logy=true, file="plots/convergence_sw", families_shown=["baseline", "saas_map", "dsp_map", "saas_bi"],
    rows=[("axis-aligned", true, problems), ("rotated", false, problems)], ylims_log=(0.02, 4), dashed_caps=true, title="Posterior quality vs. simulation budget (median over seeds, band = min–max)")
    ncols = maximum(length(r[3]) for r in rows)
    fig = Figure(; size = (max(320 * ncols + 60, 1000), 320 * length(rows)), backgroundcolor = SURFACE, figure_padding = (16, 16, 12, 12))
    present = unique(method_family(r.key[2]) for r in results)
    families = filter(in(present), families_shown)
    for (i, (label, aligned, probs)) in enumerate(rows), (j, prob) in enumerate(probs)
        ax = Axis(fig[i + 1, j];
            title = "$(replace(prob, r"^wavy_|_emb$|_sum$" => ""))$(get(RELEVANT_DIMS, prob, "")) – $label",
            xlabel = i == length(rows) ? "simulations" : "",
            xscale = log10, xticks = ([20, 30, 50, 100, 200], ["20", "30", "50", "100", "200"]),
            yscale = logy ? log10 : identity,
            yticks = logy ? ([0.01, 0.03, 0.1, 0.3, 1, 3], ["0.01", "0.03", "0.1", "0.3", "1", "3"]) : Makie.automatic,
        )
        logy ? ylims!(ax, ylims_log...) : ylims!(ax, -0.05, 1.05)
        style_axis!(ax)
        for fam in families
            rs = [r for r in results if problem_base(r.key[1]) == prob && is_aligned(r.key[1]) == aligned && method_family(r.key[2]) == fam && !isempty(r.scores)]
            isempty(rs) && continue
            ns, med, lo, hi = aggregate(rs, field)
            isempty(ns) && continue
            band!(ax, ns, lo, hi; color = (COLORS[fam], 0.16))
            lines!(ax, ns, med; color = COLORS[fam], linewidth = 2, linestyle = dashed_caps && endswith(fam, "_cap") ? :dash : :solid)
        end
        xlims!(ax, 18, 215)
    end
    Label(fig[2:length(rows)+1, 0], ylabel; rotation = pi / 2, color = INK2, fontsize = 12)
    elems = [LineElement(color = COLORS[f], linewidth = 2, linestyle = dashed_caps && endswith(f, "_cap") ? :dash : :solid) for f in families]
    Legend(fig[1, 1:ncols], elems, [LABELS[f] for f in families];
        orientation = :horizontal, nbanks = ncols == 1 ? 2 : (length(families) > 5 ? 2 : 1), framevisible = false, labelcolor = INK2, labelsize = 12, halign = :left)
    Label(fig[0, 1:ncols], title; color = INK, fontsize = 15, halign = :left, font = :bold)
    mkpath(dirname(file))
    save(file * ".png", fig; px_per_unit = 2)
    save(file * ".pdf", fig)
    return fig
end

const LATENT_FAMILIES = ["baseline", "dsp_map", "latent_k3", "latent_ktrue", "oracle_gp"]
plot_latent(; kwargs...) = (rs = load_results(); plot_curves(rs; families_shown = LATENT_FAMILIES, file = "plots/convergence_latent_sw", kwargs...))

const ALL_FAMILIES = ["baseline", "saas_map", "dsp_map", "latent_k3", "latent_ktrue", "oracle_gp"]
plot_all_recall(; kwargs...) = (rs = load_results(); plot_curves(rs; field = :mode_recall, problems = ["wavy_multimodal"], families_shown = ALL_FAMILIES,
    ylabel = "mode recall (higher is better)", logy = false, file = "plots/convergence_all_recall", kwargs...))

plot_latent_recall(; kwargs...) = (rs = load_results(); plot_curves(rs; field = :mode_recall, problems = ["wavy_multimodal"], families_shown = LATENT_FAMILIES,
    ylabel = "mode recall (higher is better)", logy = false, file = "plots/convergence_latent_recall", kwargs...))

plot_all(; kwargs...) = (rs = load_results(); (
    plot_curves(rs; kwargs...);
    plot_curves(rs; field = :mode_recall, problems = ["wavy_multimodal"], ylabel = "mode recall (higher is better)", logy = false, file = "plots/convergence_recall", kwargs...)))

const PAPER_NAMES = ["ab", "simple", "banana", "bimodal"]
"Paper problems in 10D: only 2 dimensions relevant (axis-aligned, rotated) or all 10 relevant (additive)."
plot_paper_problems(; kwargs...) = (rs = load_results(); plot_curves(rs;
    families_shown = ALL_FAMILIES, ylims_log = (0.01, 4), file = "plots/convergence_paper_problems",
    rows = [("2 of 10 dims relevant, aligned", true, PAPER_NAMES .* "_emb"), ("2 of 10 dims relevant, rotated", false, PAPER_NAMES .* "_emb"), ("all 10 dims relevant, additive", true, PAPER_NAMES .* "_sum")],
    title = "Paper problems in 10D (top rows: 2 relevant dims, bottom row: all 10 relevant): posterior quality vs. budget (median, band = min–max)", kwargs...))

const EVERY_FAMILY = ["baseline", "baseline_cap", "saas_map", "saas_map_cap", "dsp_map", "dsp_map_cap", "latent_k3", "latent_ktrue", "oracle_gp"]
"All paper problems and all methods (capped dashed) in one figure."
plot_paper_all(; kwargs...) = plot_paper_problems(; families_shown = EVERY_FAMILY, file = "plots/convergence_paper_all", kwargs...)

const SIX_PROBLEMS = [("ab_emb", "ab"), ("simple_emb", "simple"), ("banana_emb", "banana"), ("bimodal_emb", "bimodal"), ("wavy_ridge", "ridge"), ("wavy_multimodal", "multimodal")]
"""
All six problems (4 paper problems, wavy ridge and multimodal) and all methods in one figure.
Additive (all 10 dims relevant) variants exist for the paper problems only.
"""
plot_six(; kwargs...) = (rs = load_results(); plot_curves(rs;
    families_shown = EVERY_FAMILY, ylims_log = (0.01, 4), file = "plots/convergence_six_problems",
    rows = [("aligned", true, first.(SIX_PROBLEMS)), ("rotated", false, first.(SIX_PROBLEMS)), ("additive, all 10 relevant", true, PAPER_NAMES .* "_sum")],
    title = "Six problems in 10D: posterior quality vs. budget (median, band = min–max; dashed = λ ≤ 5×width)", kwargs...))

const CAP_FAMILIES =["baseline", "baseline_cap", "saas_map", "saas_map_cap", "dsp_map", "dsp_map_cap", "oracle_gp"]
"Paper problems: lengthscale-capped (dashed) vs. uncapped (solid) baseline, SAAS and DSP priors."
plot_capped(; kwargs...) = plot_paper_problems(; families_shown = CAP_FAMILIES, file = "plots/convergence_paper_capped", kwargs...)

const TRUNCATED_FAMILIES = ["baseline_cap", "saas_map_cap", "dsp_map_cap", "latent_k3", "latent_ktrue", "oracle_gp"]
"Six problems with the lengthscale-truncated variants only."
plot_six_truncated(; kwargs...) = plot_six(; families_shown = TRUNCATED_FAMILIES, file = "plots/convergence_six_truncated", dashed_caps = false,
    title = "Six problems in 10D: posterior quality vs. budget (median, band = min–max; GP lengthscale priors truncated at 5×width)", kwargs...)
