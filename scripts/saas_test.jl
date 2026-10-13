include("../src/include_code.jl")

function saas_test()
    d1 = SaasPrior([1.])
    s = range(-14, 14; length=4001)
    ds = step(s)
    ps = [pdf(d1, [exp(v)]) * exp(v) for v in s]
    @info "1D normalization" sum(ps) * ds
    m_quad = sum(ps .* s) * ds
    m_mc = mean(log(rand(d1)[1]) for _ in 1:200_000)
    @info "1D E[log ℓ]" m_quad m_mc

    d2 = SaasPrior([1., 1.])
    s2 = range(-10, 10; length=501)
    mass = sum(pdf(d2, [exp(a), exp(b)]) * exp(a) * exp(b) for a in s2, b in s2) * step(s2)^2
    @info "2D normalization (truncated box)" mass
    P = [pdf(d2, [exp(a), exp(b)]) * exp(a + b) for a in s2, b in s2] .* step(s2)^2
    p_quad = sum(P[(s2 .< 0), (s2 .< 0)]) 
    xs = rand(d2, 200_000)
    p_mc = mean((xs[1, :] .< 1) .& (xs[2, :] .< 1))
    @info "2D P(ℓ1<1, ℓ2<1)" p_quad p_mc

    d10 = SaasPrior(fill(10., 10))
    x = rand(d10)
    @info "10D logpdf finite, ForwardDiff gradient" logpdf(d10, x) sum(abs, BOSS.ForwardDiff.gradient(y -> logpdf(d10, y), x))
end
