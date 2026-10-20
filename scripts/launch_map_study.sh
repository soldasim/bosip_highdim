#!/usr/bin/env bash
# Usage: launch_map_study.sh [workers] "n1,n2,..."  -- MAP fits for k = 1..d on all problems, in a tmux session with an attached window
cd "$(dirname "$0")/.."
tmux new-session -d -s mapstudy -c "$PWD"
tmux send-keys -t mapstudy "julia --project=. -t ${1:-2} -e 'include(\"scripts/ml_study.jl\"); ml_study(; ns=[${2:-20,50,100}], workers=${1:-2})' 2>&1 | grep -a -v -i 'BOBYQA\|HONOUR\|^anged\|^bounds\|^\$'" Enter
osascript -e 'tell application "Terminal" to do script "tmux attach -t mapstudy ; tmux kill-session -t mapstudy 2>/dev/null"' -e 'tell application "Terminal" to activate'
