#!/usr/bin/env bash
# Usage: launch_capped.sh  -- starts 3 worker tmux sessions for the lengthscale-capped baseline/SAAS/DSP runs
cd "$(dirname "$0")/.."
M='[Capped(Baseline(), 5.), Capped(SaasMAP(), 5.), Capped(DimScaledMAP(), 5.)]'
launch() { # session name, problems
  tmux new-session -d -s "$1" -c "$PWD"
  tmux send-keys -t "$1" "julia --project=. -t 1 -e 'include(\"scripts/pilot.jl\"); pilot(; problems=$2, methods=$M, seeds=1:3, file=\"data/pilot_curves_paper_$1.jld2\")' 2>&1 | grep -a -v -i 'BOBYQA\|HONOUR\|^anged\|^bounds\|^\$'" Enter
  osascript -e "tell application \"Terminal\" to do script \"tmux attach -t $1 ; tmux kill-session -t $1 2>/dev/null\"" >/dev/null
}
launch pp-cap-1 '[("ab_emb", false), ("ab_emb", true), ("simple_emb", false), ("simple_emb", true)]'
launch pp-cap-2 '[("banana_emb", false), ("banana_emb", true), ("bimodal_emb", false), ("bimodal_emb", true)]'
launch pp-cap-3 '[("ab_sum", false), ("simple_sum", false), ("banana_sum", false), ("bimodal_sum", false)]'
