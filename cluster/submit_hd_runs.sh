#!/bin/bash

# ASSUMES pwd == bosip_highdim and that the reference posteriors exist.
#
# ARGS[1] = problem name.
# ARGS[2] = seeds, e.g. "1 2 3".
# ARGS[3] = methods, a subset of "dsp k3 ktrue".
# One job per (method, seed).

method_expr() {
    case $1 in
        dsp) echo "Capped(DimScaledMAP(), 5.)";;
        k3) echo "LinearEncoder(; k=3)";;
        ktrue) echo "LinearEncoder(; k=nothing)";;
    esac
}

name=$1
for m in $3; do
    for seed in $2; do
        sbatch -p cpu --cpus-per-task=1 --mem=8G --job-name="hd_${name}_${m}_${seed}" -o "logs/%x_%j.out" cluster/run_julia.sh \
            "include(\"scripts/pilot.jl\"); pilot(; problems=[(\"$name\", true)], methods=[$(method_expr $m)], seeds=[$seed], budget=203, file=\"data/hd_${name}_${m}_${seed}.jld2\")"
    done
done
