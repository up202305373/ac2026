# The final regular season ranking of teams

- **Target:** `Teams.rank`: the team's final position **inside its division** (1–5). Divisional ranking is the headline target.
- **Secondary check:** league-wide position and ranking metrics, reported by points order and by a reconstructed seed-style order.
- **Running from repository root:** `.venv/bin/python tasks/a_ranking/main.py`
- **Dependencies:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`,`seaborn`. 
- **SQL Database:** built from `data/raw/*.csv` with `src/build_db.py` if it is missing.

Each run loops over `TEST_YEARS`, re-samples `data/train` and `data/test` through `src.sampler.sampler(1995, train_until=year-1, test_year=year)`, fits every experiment, and appends the per-year predictions and metrics to `tasks/a_ranking/logs/output.log`. Aggregate summaries and plots are written at the end of the run.

## Test years

`TEST_YEARS = [2000, 2001, 2002, 2003, 2006, 2007, 2008, 2009]`.

2004 is absent from the data entirely: that season was cancelled by the lockout, so there are no standings to predict.

2005 has standings but is excluded from `TEST_YEARS`: with 2004 missing, its prior-season features come from 2003, two seasons back, unlike every other fold. We predicted it would be an outlier so it was evaluated separately as a robustness check. Divisional accuracy was in line with or above the 8-year averages, but points MAE rose by roughly two points (11.64 vs 9.40) for `roster_all` and by a similar amount for the other models. Adding 2005 raises its nine-year points MAE to 9.65 and barely changes the divisional rate (101/270 vs 88/240). It is therefore kept out of the headline results.

## Metrics

`evaluate.evaluate_ranking()` reports divisional metrics first:

- **divisional positions correct / total**
- **division winners correct / total**
- **mean absolute divisional position error**
- **divisional Spearman correlation**

League-wide metrics are secondary and come in two views:

- **point order:** all teams ranked by points; reports exact positions, mean absolute position error and Spearman (`league_pts_spearman`).
- **seed style:** division winners are placed before other teams inside each conference, each group is ordered by points, and the conference lists are interleaved `E1, W1, E2, W2, ...` to make one 1–30 list. This reports exact positions, mean absolute position error and `league_seed_spearman`.

For the test years (2000–2009), the relevant NHL rule was: the three division winners in each conference were conference seeds 1–3 by points, and the next five teams by points were seeds 4–8. The database has no seed column, so the same convention is extended to all 15 teams per conference. Ties use wins, goal differential and goals for for actual standings; predicted points are effectively continuous, so predicted ties fall back to `franchID`.

Points MAE and RMSE are also reported for the underlying regression target.

## Models

Models are registered in `models/__init__.py` and selected by name in `config.EXPERIMENTS`:

| name | prediction |
|---|---|
| `baseline` | carry the latest training season's points forward |
| `roster_base` | ridge on base roster features |
| `roster_all` | base + shots + special-team features |
| `roster_shots` / `roster_special` / `roster_pts` | base + one extra group |
| `roster_all_pts` | base + shots + special + prior points |
| `roster_base_age` / `roster_all_age` | same models plus roster-age features |

The `roster_*` models are `roster.train_and_predict` with different feature groups. Each group is a set of columns in `models/roster.py`:

- **base**: `team_gd`, `fwd_ppg`, `goalie_svpct`, `new_share`
- **shots**: `shot_diff`, `shoot_pct`, `team_sv_pct`
- **special**: `pp_pct`, `pk_pct`
- **prior_pts**: `prior_pts`
- **age**: `age_mean`, `age_std`, `age_veteran_share` (from `Master.birthYear`)

Roster features are built from the target season's known-before-the-season roster plus the most recent prior season(s), then standardised and fitted with `Ridge`. When `alpha` is not fixed, it is selected on a held-out training season (the latest one available before the target year) by points MAE, and the model is then refit on all training seasons. The chosen value is logged. This keeps validation strictly before the test year.

## Configuration options

Everything that gets tested lives in `tasks/a_ranking/config.py`:

- `TEST_YEARS` – years to evaluate on.
- `ALPHAS` – candidate ridge penalties for validation.
- `EXPERIMENTS` – list of plain dicts. Each has a `name` (log label), a `model` (key in `MODELS`), and optional overrides:
  - `validation_alphas` – penalties to try; omit for a fixed `alpha`.
  - `previous_years` – how many prior seasons to average features over (default 1). Lookback 2 and 3 are included as experiments.
  - `n_forwards` – top forwards used for `fwd_ppg` (default 12; `3`, `6`, and `"all"` are swept).
  - `min_skater_gp` / `min_goalie_gp` – minimum prior-season games for a skater / goalie to count (defaults 20 / 10; `40` / `20` are swept).
  - `train_from` – first training season passed to the sampler (default 1995; `1980`, `1985`, `1990`, `2000` are swept). Years with fewer than two usable training seasons are skipped for that experiment.
  - `use_age` – enable the `age` feature group from `Master` (only meaningful with the `roster_*_age` models).
  - `estimator` – `"ridge"` (default), `"elasticnet"` or `"ols"`. ElasticNet reuses the alpha validation grid and adds `l1_ratio` (default `0.5`); OLS is unregularized, so it uses a fixed `alpha: 0` and skips validation. Both reuse the existing `roster_base` / `roster_all` feature sets.

Add new experiments by appending another dict; no other file needs to change unless it introduces a genuinely new model or feature group.

## Outputs

- `tasks/a_ranking/logs/output.log` – per-year predictions and metrics, plus an aggregate summary for each experiment with sd for every metric and a compact per-year table. Log entries use foldable indentation so each run and model block can be collapsed in an editor.
- `tasks/a_ranking/plots/` – six curated figures written once by `evaluate.generate_graphs()` at the end of a run: `model_comparison.png` (headline models × divisional metrics, mean ± sd), `league_comparison.png` (point-order, seed-style and Spearman metrics), `per_year_trends.png`, `error_distributions.png` (league-wide and divisional, `roster_base` vs `roster_all`), `predicted_vs_actual.png` and `estimator_comparison.png` (Ridge vs ElasticNet vs OLS for both feature sets).
