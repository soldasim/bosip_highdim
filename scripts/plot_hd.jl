include("plot_curves.jl")

const HD_NAMES = [("rosenbrock5_cross", "Rosenbrock5"), ("styblinski_tang5_cross", "Styblinski-Tang5"), ("michalewicz5_cross", "Michalewicz5"),
    ("sphere5_cross", "Sphere5"), ("duffing5", "Duffing5"), ("diffusion5d", "Diffusion5D")]
const HD_METHODS = [("dsp_map_cap5.0", "dsp", "DS-GP"), ("linenc_k3", "k3", "Linear encoder (capped), k=3"), ("linenc_ktrue", "ktrue", "Linear encoder (capped), k=5")]
const HD_COLORS = Dict("dsp" => COLORS["dsp_map"], "k3" => COLORS["linenc_k3"], "ktrue" => COLORS["linenc_ktrue"])
const PAPER_STYLES = [("gp", "Paper: GP (MaxVar)", :dash), ("warped_gp", "Paper: WarpedGP (MaxVar)", :dot)]

paper_tv(; file="paper_tv/paper_tv.jld2") = isfile(file) ? jldopen(f -> (f["paper"], f["n"]), file) : nothing

hd_results(dir="data") = reduce(vcat, [load(joinpath(dir, f), "results") for f in sort(readdir(dir)) if occursin(r"^hd_.*\.jld2$", f)]; init=Any[])

"TV scores `(n, tv)` of every available seed of a run set."
hd_tv(name, method; dir="data") = [load(f, "scores") for f in joinpath.(dir, name, ["$(method)_$(s)_tv.jld2" for s in 1:3]) if isfile(f)]

function band_plot!(ax, curves, color)
    ns = sort(unique(c.n for cs in curves for c in cs))
    vals = [[c.v for cs in curves for c in cs if c.n == n && !isnan(c.v)] for n in ns]
    keep = .!isempty.(vals)
    ns, vals = ns[keep], vals[keep]
    band!(ax, ns, minimum.(vals), maximum.(vals); color = (color, 0.16))
    lines!(ax, ns, median.(vals); color, linewidth = 2)
end

function plot_hd(; file="plots/convergence_hd5", results=hd_results(), hd_dir="data", paper=paper_tv())
    fig = Figure(; size = (1900, 700), backgroundcolor = SURFACE, figure_padding = (16, 16, 12, 12))
    rows = [("sliced Wasserstein (lower is better)", :sw), ("total variation (lower is better)", :tv)]
    for (i, (ylabel, kind)) in enumerate(rows), (j, (name, title)) in enumerate(HD_NAMES)
        ax = Axis(fig[i + 1, j]; title = i == 1 ? title : "", xlabel = i == 2 ? "simulations" : "", xscale = log10,
            xticks = ([3, 10, 30, 100, 200], ["3", "10", "30", "100", "200"]), yscale = kind == :sw ? log10 : identity)
        style_axis!(ax)
        for (mname, key, _) in HD_METHODS
            curves = if kind == :sw
                [[(; n = s.n, v = s.metrics.sliced_wasserstein) for s in r.scores] for r in results if r.key[1] == name && r.key[2] == mname && !isempty(r.scores)]
            else
                [[(; n = s.n, v = s.tv) for s in sc] for sc in hd_tv(name, mname; dir = hd_dir)]
            end
            isempty(curves) || band_plot!(ax, curves, HD_COLORS[key])
        end
        if kind == :tv && !isnothing(paper)
            runs, n = paper
            for (key, _, style) in PAPER_STYLES
                lines!(ax, n, vec(median(reduce(hcat, runs[name][key]); dims = 2)); color = INK2, linewidth = 1.8, linestyle = style)
            end
        end
        kind == :sw ? ylims!(ax, 0.002, 5) : ylims!(ax, -0.02, 1.05)
        xlims!(ax, 2.7, 215)
        j == 1 && (ax.ylabel = ylabel)
    end
    elems = [LineElement(color = HD_COLORS[k], linewidth = 2) for (_, k, _) in HD_METHODS]
    labels = collect(last.(HD_METHODS))
    if !isnothing(paper)
        append!(elems, [LineElement(color = INK2, linewidth = 1.8, linestyle = style) for (_, _, style) in PAPER_STYLES])
        append!(labels, [label for (_, label, _) in PAPER_STYLES])
    end
    Legend(fig[1, 1:6], elems, labels; orientation = :horizontal, framevisible = false, labelcolor = INK2, labelsize = 12, halign = :left)
    Label(fig[0, 1:6], "5D problems of the BOSIP paper: posterior quality vs. budget (median over 3 runs, band = min–max)"; color = INK, fontsize = 15, halign = :left, font = :bold)
    mkpath(dirname(file))
    save(file * ".png", fig; px_per_unit = 2)
    save(file * ".pdf", fig)
    return fig
end

"Median over seeds of the final scores (at the last checkpoint), per problem and method."
function hd_summary(; results=hd_results(), hd_dir="data")
    for (name, _) in HD_NAMES
        line = rpad(name, 24)
        for (mname, key, _) in HD_METHODS
            sw = [last(filter(!isnan, [x.metrics.sliced_wasserstein for x in r.scores])) for r in results if r.key[1] == name && r.key[2] == mname && !isempty(r.scores)]
            tv = [last(filter(!isnan, [x.tv for x in sc])) for sc in hd_tv(name, mname; dir = hd_dir)]
            line *= "  $(rpad(key, 5)) SW $(round(median(sw); digits = 3)) TV $(isempty(tv) ? "-" : round(median(tv); digits = 3))"
        end
        println(line)
    end
end
