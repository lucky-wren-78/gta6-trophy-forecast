[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lucky-wren-78/gta6-trophy-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lucky-wren-78/gta6-trophy-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecasts each Grand Theft Auto VI (PS5) trophy's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at the case's target time, as five quantiles (5%, 25%, 50%, 75%, 95%) per
trophy. Entry for the
[make.fun competition](https://make.fun/c/gta6-trophy-rarity-forecast-01).

## How it works

`main.py` (Python 3.13, standard library only, no network) forecasts
earners and tracked players separately, then combines them:

```
rate_i(target) = 100 * E_i(target) / N(target)
```

- **N(t), players tracked**: a power law in time since release,
  `log N = a + b * log(1 + t)`, fitted on the last 48 hours of distinct
  reads (launch growth slows, so the early steep slope must not leak
  into the forecast).
- **E_i(t), earners of trophy i**: the same power-law fit per trophy,
  with each slope shrunk toward its tier's mean slope (trophies of a
  tier share how fast players reach them).
- Fits run on **distinct reads only**: Exophase refreshes in steps, so
  long stretches of byte-identical reads carry no information and must
  not flatten the slopes.
- **Stale-page fallback**: when the whole history is one distinct read
  (the launch-night practice cases), no slope can be fitted. The page
  is frozen while the real player base grows, so the next refresh
  dilutes every rate. The rules state the one observed refresh (638
  players at 23:48 UTC, 3,868 at 04:30 UTC, ~6x over 4.7 h); the
  fallback assumes tracked players keep multiplying at that pace while
  earners grow only part-way toward it (early trophies faster).
- **Spread** (logit scale): a random-walk term growing with the horizon,
  the pooled residual scatter of the fits, and a staleness term for a
  frozen page. A frozen page gets a wide band (~1.1 logit): the house
  baseline's narrow 90-minute band covered ~0% of the refreshed rates.

## Validation

No practice answers are published, so the backtest (`backtest.py`) is
simulation-based: a synthetic launch week calibrated to the observed
launch-night state (638 players, the kit's practice rates, the 6x
refresh), scored against the starter persistence baseline and the house
curve baseline. Mean pinball loss (logit scale, lower is better),
averaged over 4 random seeds:

| case | this entry | persistence | curve baseline |
| --- | ---: | ---: | ---: |
| practice-like cut, target +90 min (p00) | **0.27** | 1.42 | 1.42 |
| practice-like cut, target +90 min (p01) | **0.26** | 1.50 | 1.50 |
| final-like cut, target +4.5 days (f00) | **0.20** | 0.27 | 0.25 |

Practice sensitivity check (assuming the refreshed answers equal last
rate x 0.22, the factor implied by the rules' stated baseline practice
loss of ~1.51): this entry 0.26 vs 1.40 for both baselines.

Deterministic (two runs byte-identical); runs in ~0.07 s per case,
far under the 8 s limit.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
