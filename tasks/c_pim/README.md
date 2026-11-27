# The regular season penalty minutes of players

- **Target:** `Scoring.PIM` from `stint = 1` for NHL players in the target season. This is the player's PIM with the team he starts the season with, since we won't be predicting trades.
- **Running from repository root:** `.venv/bin/python tasks/c_pim/main.py`
- **Dependencies:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.
- **SQL Database:** built from `data/raw/*.csv` with `src/build_db.py` if it is missing.

Each run loops over `TEST_YEARS`, re-samples `data/train` and `data/test` through `src.sampler.sampler(1995, train_until=year-1, test_year=year)`, fits every experiment, and appends the per-year predictions and metrics to `tasks/c_pim/logs/output.log`. Aggregate summaries are written at the end of the run.

## Test years

`TEST_YEARS = [2000, 2001, 2002, 2003, 2006, 2007, 2008, 2009]`.

2004 is absent from the data entirely: that season was cancelled by the lockout, so there are no player-season PIM totals to predict.

2005 has scoring data but is excluded: with 2004 missing, its prior-season features would come from 2003, two seasons back, unlike every other fold. It is therefore kept out of the headline results. No separate 2005 robustness check is included here.

## Metrics

`evaluate.evaluate_pim()` reports:

- **MAE** — mean absolute PIM error.
- **RMSE** — root mean squared PIM error.
- **Spearman correlation** — rank correlation between predicted and actual PIM.
- **players evaluated / total** — players with a recorded target-season PIM.

Missing predictions are scored as `0`, so no player is dropped for lacking a prediction. The baseline also treats a missing prior-season PIM as `0`. Only players with no recorded target-season PIM are excluded.

## Models

Models are registered in `models/__init__.py` and selected by name in `config.EXPERIMENTS`:

| name | prediction |
|---|---|
| `baseline` | carry the player's total PIM across all stints from the latest training season forward |
| `player_base` | Ridge regression on base prior-season player features |
| `player_base_pos` | base features plus prior-season position dummies |

The `player_*` models are `player.train_and_predict` with different feature groups. Each group is a set of columns in `models/player.py`:

- **base**: `prior_pim`, `prior_gp`, `prior_sog`, `prior_pim_per_gp`, `has_prior`
- **pos**: `pos_C`, `pos_D`, `pos_G`, `pos_L`, `pos_R`

Features are built from the most recent prior season available in training data, aggregated across all of that season's stints because those rows are observed. Players with no prior-season row get zeroed numeric features and `has_prior = 0`; `has_prior` is `1` whenever the player had any row in that prior season. Features are standardised and fitted with `Ridge`; predictions are clipped at `0`.

When `alpha` is not fixed, it is selected on a held-out training season (the latest one available before the target year) by PIM MAE, and the model is then refit on all training seasons. The chosen value is logged. This keeps validation strictly before the test year.

## Results so far

Full eight-year run:

| model | MAE | RMSE | Spearman | players scored |
|---|---:|---:|---:|---:|
| `baseline` | 18.46 | 29.91 | 0.693 | 7744/7761 |
| `player_base` | 17.84 | 26.27 | 0.701 | 7744/7761 |
| `player_base_pos` | **17.63** | **26.19** | **0.707** | 7744/7761 |

`player_base_pos` is best on all three metrics: adding position lowers MAE from `17.84` to `17.63` and raises Spearman from `0.701` to `0.707`. The selected alpha varies by year from `0.1` to `300.0` for both `player_*` models, so the validation choice is somewhat unstable; that is worth checking before adding more variants.

## Configuration options

Everything that gets tested lives in `tasks/c_pim/config.py`:

- `TEST_YEARS` — years to evaluate on.
- `ALPHAS` — candidate ridge penalties for validation.
- `EXPERIMENTS` — list of plain dicts. Each has a `name` (log label), a `model` (key in `MODELS`), and optional overrides:
  - `alphas` — penalties to try for the `player_*` models.

The `player_base` and `player_base_pos` keys in `MODELS` are `functools.partial` calls that select the `base` and `base + pos` feature groups. Add new experiments by appending another dict; no other file needs to change unless it introduces a genuinely new model or feature group.

## Known limitation: no-trade assumption

The sampler hides `stint >= 2` rows in the test season, and the target uses only `stint = 1`. The target is therefore **PIM with the team the player starts the season with**: it assumes the player is not traded. At prediction time we do not know whether a trade will happen, so this is a deliberate operationalization rather than an accidental undercount.

Prior-season inputs are different: `baseline` and the `player_*` models aggregate the latest prior season across all stints, because those rows are already observed in the training data.

Across the configured test years:

- 736 / 7,761 player-seasons have multiple stints (**9.5%**).
- Later stints add **12,564 PIM**, about **4.5%** of total league PIM.

Those numbers describe the gap relative to a season-total target. An alternative target would sum PIM across all stints in the test season, which would require predicting trades or later stints. We are intentionally not doing that here.

## Outputs

- `tasks/c_pim/logs/output.log` — per-year predictions and metrics, plus an aggregate summary for each experiment with sd for every metric.
- `tasks/c_pim/plots/` — one error histogram per experiment, currently overwritten by the latest evaluated year.
