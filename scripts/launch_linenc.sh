#!/usr/bin/env bash
# Usage: launch_linenc.sh  -- starts 4 worker tmux sessions for the LinearEncoder / OracleEncoder runs
cd "$(dirname "$0")/.."
ALL='[OracleEncoder(), LinearEncoder(; k=3), LinearEncoder(; k=nothing)]'
SUM='[LinearEncoder(; k=3), LinearEncoder(; k=nothing)]'
launch() { # session name, problems, methods
  tmux new-session -d -s "$1" -c "$PWD"
  tmux send-keys -t "$1" "julia --project=. -t 1 -e 'include(\"scripts/pilot.jl\"); pilot(; problems=$2, methods=$3, seeds=1:3, file=\"data/pilot_curves_$1.jld2\")' 2>&1 | grep -a -v -i 'BOBYQA\|HONOUR\|^anged\|^bounds\|^\$'" Enter
}
launch linenc-1 '[("ab_emb", false), ("ab_emb", true), ("simple_emb", false), ("simple_emb", true)]' "$ALL"
launch linenc-2 '[("banana_emb", false), ("banana_emb", true), ("bimodal_emb", false), ("bimodal_emb", true)]' "$ALL"
launch linenc-3 '[("wavy_ridge", false), ("wavy_ridge", true), ("wavy_multimodal", false), ("wavy_multimodal", true)]' "$ALL"
launch linenc-4 '[("ab_sum", false), ("simple_sum", false), ("banana_sum", false), ("bimodal_sum", false)]' "$SUM"
