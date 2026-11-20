# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wiliamkesh/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wiliamkesh/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every Grand Theft Auto VI trophy on PlayStation 5, its
earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per
trophy — a [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty entry.

## Model

Two-process cohort hazard & launch-dilution (see [MODEL.md](MODEL.md)):
the page shows a ratio of two integer counts — players tracked `P(t)`
and per-trophy earners `E_i(t)` — that only move when Exophase
refreshes. The entry models the two counts separately in log-time,
on a recent 24-hour window of distinct refresh states, with
empirical-Bayes tier shrinkage on the earner slopes, a rolling-origin
backtest choosing between power-law and saturating players growth,
and a conformal rolling-origin spread scaled to the final horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, no network,
deterministic: two runs on the same case write byte-identical output.
It reads every `*.in` in the input directory and writes
`<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Validation

Launch-week simulations (synthetic cases, cut day 2.5, target day 7):
pinball loss on the logit scale 0.063-0.064 in power-law growth
scenarios and 0.25-0.27 when players growth saturates, beating the
persistence baseline (0.49-0.66) and the trend baseline (0.23-0.42)
in all 8 runs. Details and limitations in [MODEL.md](MODEL.md).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
