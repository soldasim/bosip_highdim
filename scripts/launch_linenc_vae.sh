#!/usr/bin/env bash
# Usage: launch_linenc_vae.sh  -- starts 4 worker tmux sessions (with attached windows) for the VAE-penalised LinearEncoder runs
cd "$(dirname "$0")/.."
METHODS='[LinearEncoder(; k=3, vae=0.5), LinearEncoder(; k=nothing, vae=0.5)]'
launch() { # session name, problems
  tmux new-session -d -s "$1" -c "$PWD"
  tmux send-keys -t "$1" "julia --project=. -t 1 -e 'include(\"scripts/pilot.jl\"); pilot(; problems=$2, methods=$METHODS, seeds=1:3, file=\"data/pilot_curves_$1.jld2\")' 2>&1 | grep -a -v -i 'BOBYQA\|HONOUR\|^anged\|^bounds\|^\$'" Enter
  osascript -e "tell application \"Terminal\" to do script \"tmux attach -t $1 ; tmux kill-session -t $1 2>/dev/null\""
}
launch vae-1 '[("ab_emb", false), ("ab_emb", true), ("simple_emb", false), ("simple_emb", true)]'
launch vae-2 '[("banana_emb", false), ("banana_emb", true), ("bimodal_emb", false), ("bimodal_emb", true)]'
launch vae-3 '[("wavy_ridge", false), ("wavy_ridge", true), ("wavy_multimodal", false), ("wavy_multimodal", true)]'
launch vae-4 '[("ab_sum", false), ("simple_sum", false), ("banana_sum", false), ("bimodal_sum", false)]'
osascript -e 'tell application "Terminal" to activate'
