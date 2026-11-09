using Pkg

Pkg.rm(["BOSS", "BOSIP", "BIPBM"])
Pkg.develop(path = expanduser("~/repos/highdim_deps/BOSS.jl"))
Pkg.develop(path = expanduser("~/repos/highdim_deps/BOSIP.jl"))
Pkg.develop(path = expanduser("~/repos/highdim_deps/BIPBM.jl"))
Pkg.add("DifferentialEquations")
Pkg.instantiate()
Pkg.precompile()
Pkg.status()
