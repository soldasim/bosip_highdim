# high-dim-bosip

BOSIP experiments in 10D with about 200 simulations, where only a few input dimensions affect the likelihood.
The question is whether low-dimensional or latent GP surrogates and better GP lengthscale priors recover the posterior (all modes and the rough shape).

## Setup

Built on [BOSS.jl](https://github.com/soldasim/BOSS.jl) and [BOSIP.jl](https://github.com/soldasim/BOSIP.jl), both taken from their `dev` branches (see `[sources]` in `Project.toml`).

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Problems

- `src/problem.jl`: `LatentProblem`. The likelihood depends only on the first `k` coordinates of `Q x` (`Q` = identity for axis-aligned, a random rotation otherwise); the other coordinates are prior-dominated. Domain is `[-5, 5]^10`.
- `src/benchmarks.jl`:
  - `wavy_ridge` (k=1), `wavy_banana`, `wavy_multimodal` (k=2).
  - The four analytical problems of the BOSIP paper (`ab`, `simple`, `banana`, `bimodal`) as `<name>_emb` (2 of 10 dims relevant) and `<name>_sum` (additive over 5 coordinate pairs, all 10 relevant, axis-aligned only).

## Methods (`src/methods.jl`)

- `Baseline`: default BOSIP GP (inverse-gamma lengthscale priors).
- `SaasMAP`, `DimScaledMAP`: SAAS and dimension-scaled lengthscale priors, fitted by MAP. `SaasBI`: SAAS fitted by NUTS.
- `Capped(method, 5.)`: truncates the lengthscale priors of `method` at 5 domain widths. Use the capped variants; uncapped fits reach lengthscales of 10³–10⁶ widths, which are not identifiable and break the final GP variance computation.
- `LatentGP` (`src/latent_gp.jl`): GP on a learned linear encoder `z = W x` with `k` latent dimensions (`k=3` or the true `k`).
- `OracleGP`: GP on the true informed subspace (additive kernel for `_sum` problems).

New variants: subtype `Method` and define `setup(::Method, problem)`.

## Running

- `scripts/run.jl <problem> <d> <seed> [budget]`: a single run.
- `scripts/pilot.jl`: `pilot(; problems, methods, seeds, ...)` runs a grid, records scores every 5 simulations and resumes from the output file. Failed runs go to `*_failures.txt`.
- `scripts/launch_paper.sh`, `scripts/launch_capped.sh`: tmux workers for the paper-problem grids.
- `src/run.jl`: `run_experiment`. If the final sampling fails, the lengthscales are clamped at 5 widths and the run is marked `clamped`. Saves data, per-iteration fitted parameters, scores and final samples to `data/<problem>/`.
- `scripts/resample_failed.jl`: re-samples failed runs from their saved parameters with clamped lengthscales.

## Metrics and plots

- `src/metrics.jl`: reference posterior samples (prior importance sampling), sliced Wasserstein-1 and mode recall, both in the informed coordinates.
- `scripts/summarize.jl`: result table.
- `scripts/plot_curves.jl`: convergence plots (`plot_six_truncated()`).
- `scripts/plot_posteriors.jl`: final posterior samples vs. the true posterior (`plot_truncated_posteriors()`).

`data/` and `plots/` are not tracked.
