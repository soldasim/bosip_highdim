#!/bin/bash

# ASSUMES pwd == bosip_highdim
#
# ARGS[1] = a Julia expression to run with the project environment, with SLURM_CPUS_PER_TASK threads.

~/.juliaup/bin/julialauncher --project=. -t "${SLURM_CPUS_PER_TASK:-1}" -e "$1"
