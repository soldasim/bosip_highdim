#!/bin/bash

# ASSUMES pwd == bosip_highdim and that the runs of the problem have finished.
#
# ARGS[1] = problem name.
# ARGS[2] = seeds, e.g. "1 2 3".
# ARGS[3] = methods, a subset of "dsp k3 ktrue".
# One job per (method, seed); skips runs whose TV file exists.

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
        sbatch -p cpu --cpus-per-task=1 --mem=8G --job-name="hdtv_${name}_${m}_${seed}" -o "logs/%x_%j.out" cluster/run_julia.sh \
            "include(\"scripts/hd_tv.jl\"); hd_tv(\"$name\", $(method_expr $m); seeds=[$seed], every=1, check=false)"
    done
done
