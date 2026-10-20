#!/bin/bash

# ASSUMES pwd == bosip_highdim
#
# Computes the reference posteriors of the six 5D problems. Optional ARGS: the problem names.

problems=("${@:-rosenbrock5_cross styblinski_tang5_cross michalewicz5_cross sphere5_cross duffing5 diffusion5d}")
for name in ${problems[@]}; do
    sbatch -p cpu --cpus-per-task=16 --mem=32G --job-name="ref_$name" -o "logs/%x_%j.out" \
        cluster/run_julia.sh "include(\"scripts/hd_reference.jl\"); hd_reference(\"$name\")"
done
