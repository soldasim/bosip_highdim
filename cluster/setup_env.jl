using Pkg

Pkg.rm(["BOSS", "BOSIP"])
Pkg.develop(path = expanduser("~/repos/highdim_deps/BOSS.jl"))
Pkg.develop(path = expanduser("~/repos/highdim_deps/BOSIP.jl"))
Pkg.add("DifferentialEquations")
Pkg.instantiate()
Pkg.precompile()
Pkg.status()
