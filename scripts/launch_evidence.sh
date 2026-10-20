#!/usr/bin/env bash
# Usage: launch_evidence.sh [threads]  -- runs the offline evidence study in a tmux session with an attached window
cd "$(dirname "$0")/.."
tmux new-session -d -s evidence -c "$PWD"
tmux send-keys -t evidence "julia --project=. -t ${1:-10} -e 'include(\"scripts/evidence_study.jl\"); evidence_study()' 2>&1 | grep -a -v -i 'BOBYQA\|HONOUR\|^anged\|^bounds\|^\$'" Enter
osascript -e 'tell application "Terminal" to do script "tmux attach -t evidence ; tmux kill-session -t evidence 2>/dev/null"' -e 'tell application "Terminal" to activate'
