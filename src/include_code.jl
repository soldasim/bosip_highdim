using BOSS
using BOSIP
using AbstractGPs
using Distributions
using KernelFunctions
using LinearAlgebra
using OptimizationPRIMA
using Statistics
using Random
using JLD2
using Turing
import BIPBM

include("problem.jl")
include("benchmarks.jl")
if isnothing(Base.find_package("DifferentialEquations"))
    @warn "DifferentialEquations is not installed; skipping the 5D problems of `hd_problems.jl`."
else
    include("hd_problems.jl")
end
include("bipbm.jl")
include("metrics.jl")
include("methods/saas.jl")
include("methods/latent_gp.jl")
include("methods/projected_gp.jl")
include("methods/methods.jl")
include("run.jl")
include("evidence.jl")
include("reference_smc.jl")
