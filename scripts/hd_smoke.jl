include("../src/include_code.jl")

for name in sort(collect(keys(HD_PROBLEMS)))
    p = PROBLEMS[name](5)
    X = rand(prior(p), 8)
    t = @elapsed (ll = threaded_loglike(p, X))
    starts = [start_points(p, s) for s in 1:3]
    lo, hi = extrema(reduce(hcat, starts))
    v_true = to_unit(p.box[1] .* 0 .+ 0.0, p.box...)  # placeholder to exercise to_unit
    @info "Problem" name y_size = y_size(p) range_of_starts = (lo, hi) time_per_loglike = t / 8 loglike_range = extrema(ll) amp = unique(p.amp_est)
    @assert all(-5 .<= X .<= 5) && -5 <= lo && hi <= 5
end
