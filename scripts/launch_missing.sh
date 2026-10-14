#!/usr/bin/env bash
# Usage: launch_missing.sh  -- starts 7 worker tmux sessions for the runs missing from the plotted figures:
# truncated baseline/SAAS/DSP on wavy ridge/multimodal, k=true encoder on the additive problems, rotated ab oracle (seed 1).
cd "$(dirname "$0")/.."
TRUNC='[Capped(Baseline(), 5.), Capped(SaasMAP(), 5.), Capped(DimScaledMAP(), 5.)]'
launch() { # session name, output file, problems, methods, seeds
  tmux new-session -d -s "$1" -c "$PWD"
  tmux send-keys -t "$1" "julia --project=. -t 1 -e 'include(\"scripts/pilot.jl\"); pilot(; problems=$3, methods=$4, seeds=$5, file=\"$2\")' 2>&1 | grep -a -v -i 'BOBYQA\|HONOUR\|^anged\|^bounds\|^\$'" Enter
}
launch wavy-ridge-al  data/pilot_curves_trunc_wavy-ridge-al.jld2  '[("wavy_ridge", false)]'      "$TRUNC" 1:3
launch wavy-ridge-rot data/pilot_curves_trunc_wavy-ridge-rot.jld2 '[("wavy_ridge", true)]'       "$TRUNC" 1:3
launch wavy-multi-al  data/pilot_curves_trunc_wavy-multi-al.jld2  '[("wavy_multimodal", false)]' "$TRUNC" 1:3
launch wavy-multi-rot data/pilot_curves_trunc_wavy-multi-rot.jld2 '[("wavy_multimodal", true)]'  "$TRUNC" 1:3
launch ktrue-sum-1 data/pilot_curves_trunc_ktrue-sum-1.jld2 '[("ab_sum", false), ("simple_sum", false)]'     '[LatentGP(; k=nothing)]' 1:3
launch ktrue-sum-2 data/pilot_curves_trunc_ktrue-sum-2.jld2 '[("banana_sum", false), ("bimodal_sum", false)]' '[LatentGP(; k=nothing)]' 1:3
launch oracle-ab-rot data/pilot_curves_paper_pp-ab-emb.jld2 '[("ab_emb", true)]' '[OracleGP()]' 1:1
