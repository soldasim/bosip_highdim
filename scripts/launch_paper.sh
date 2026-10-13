#!/usr/bin/env bash
# Usage: launch_paper.sh  -- starts 6 worker tmux sessions for the paper-problem grid
cd "$(dirname "$0")/.."
EMB='[OracleGP(), Baseline(), SaasMAP(), DimScaledMAP(), LatentGP(), LatentGP(; k=nothing)]'
SUM='[OracleGP(), Baseline(), SaasMAP(), DimScaledMAP(), LatentGP()]'
launch() { # session name, problems, methods
  tmux new-session -d -s "$1" -c "$PWD"
  tmux send-keys -t "$1" "julia --project=. -t 1 -e 'include(\"scripts/pilot.jl\"); pilot(; problems=$2, methods=$3, seeds=1:3, file=\"data/pilot_curves_paper_$1.jld2\")' 2>&1 | grep -a -v -i 'BOBYQA\|HONOUR\|^anged\|^bounds\|^\$'" Enter
  osascript -e "tell application \"Terminal\" to do script \"tmux attach -t $1 ; tmux kill-session -t $1 2>/dev/null\"" >/dev/null
}
launch pp-ab-emb      '[("ab_emb", false), ("ab_emb", true)]' "$EMB"
launch pp-simple-emb  '[("simple_emb", false), ("simple_emb", true)]' "$EMB"
launch pp-banana-emb  '[("banana_emb", false), ("banana_emb", true)]' "$EMB"
launch pp-bimodal-emb '[("bimodal_emb", false), ("bimodal_emb", true)]' "$EMB"
launch pp-sum-1       '[("ab_sum", false), ("simple_sum", false)]' "$SUM"
launch pp-sum-2       '[("banana_sum", false), ("bimodal_sum", false)]' "$SUM"
