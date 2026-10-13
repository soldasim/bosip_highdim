include("../src/include_code.jl")
include("plot_curves.jl")

"Informed-coordinate view of the multimodal problem: the 1D map `h` and the 2D posterior density."
function plot_multimodal(; file="plots/problem_wavy_multimodal")
    p = PROBLEMS["wavy_multimodal"](10; rotate=false)
    h(v) = v^2 + 2 * sin(2 * v)
    vs = range(-5, 5; length=400)
    gs = range(-3.5, 3.5; length=700)
    prior_logpdf(v) = sum(logpdf.(Normal(0., HALF_WIDTH / 3), v))
    logpost = [true_loglike(p, vcat([a, b], zeros(p.d - 2))[:, :])[1] + prior_logpdf([a, b]) for a in gs, b in gs]
    dens = exp.(logpost .- maximum(logpost)) .^ 0.3

    fig = Figure(; size=(1000, 480), backgroundcolor=SURFACE, figure_padding=(16, 16, 12, 12))
    ax1 = Axis(fig[1, 1]; title="Output per informed coordinate", xlabel="v   (dashed: observed z = 4, band: ±2σ)", ylabel="h(v) = v² + 2 sin(2v)")
    style_axis!(ax1)
    z, s = p.z_obs[1], p.std_obs[1]
    band!(ax1, [-5, 5], fill(z - 2s, 2), fill(z + 2s, 2); color=("#2a78d6", 0.18))
    hlines!(ax1, [z]; color="#2a78d6", linewidth=1, linestyle=:dash)
    lines!(ax1, vs, h.(vs); color=INK, linewidth=2)
    xs = unique(round.(vec(p.mode_centers); digits=6))
    scatter!(ax1, xs, h.(xs); color="#eb6834", markersize=10, strokecolor=SURFACE, strokewidth=2)
    ylims!(ax1, -3, 30)

    ax2 = Axis(fig[1, 2]; title="Posterior in the informed coordinates", xlabel="v₁", ylabel="v₂", aspect=DataAspect())
    style_axis!(ax2)
    hm = heatmap!(ax2, gs, gs, dens; colormap=cgrad([SURFACE, "#2a78d6"]), colorrange=(0, 1))
    scatter!(ax2, p.mode_centers[1, :], p.mode_centers[2, :]; color="#eb6834", markersize=10, strokecolor=SURFACE, strokewidth=2)
    xlims!(ax2, -3.5, 3.5); ylims!(ax2, -3.5, 3.5)
    Colorbar(fig[1, 3], hm; label="density^0.3 (relative to max)", labelcolor=INK2, ticklabelcolor=INK2)

    Label(fig[0, 1:3], "Wavy multimodal problem: 4 modes in 2 informed directions (8 further directions are uninformed)";
        color=INK, fontsize=15, halign=:left, font=:bold)
    mkpath(dirname(file))
    save(file * ".png", fig; px_per_unit=2); save(file * ".pdf", fig)
    return fig
end
