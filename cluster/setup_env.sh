#!/bin/bash
#SBATCH -p cpufast
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=hd_setup
#SBATCH --output=logs/setup_%j.out

# ASSUMES pwd == bosip_highdim
~/.juliaup/bin/julialauncher --project=. cluster/setup_env.jl
