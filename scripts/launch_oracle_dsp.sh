#!/usr/bin/env bash
# Usage: launch_oracle_dsp.sh  -- starts 4 worker tmux sessions for the OracleDSP runs
cd "$(dirname "$0")/.."
launch() { # session name, problems
  tmux new-session -d -s "$1" -c "$PWD"
  tmux send-keys -t "$1" "julia --project=. -t 1 -e 'include(\"scripts/pilot.jl\"); pilot(; problems=$2, methods=[OracleDSP()], seeds=1:3, file=\"data/pilot_curves_oracle_dsp-${1#oracle-}.jld2\")' 2>&1 | grep -a -v -i 'BOBYQA\|HONOUR\|^anged\|^bounds\|^\$'" Enter
}
launch oracle-1 '[("wavy_multimodal", false), ("ab_emb", true), ("banana_emb", false)]'
launch oracle-2 '[("wavy_multimodal", true), ("ab_emb", false), ("banana_emb", true)]'
launch oracle-3 '[("simple_emb", false), ("bimodal_emb", true), ("wavy_ridge", true)]'
launch oracle-4 '[("bimodal_emb", false), ("simple_emb", true), ("wavy_ridge", false)]'
