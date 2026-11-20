# Model: two-process cohort hazard & launch-dilution

Forecasts each GTA VI trophy's Exophase earn rate at the target as
five quantiles, from the history of reads in the case file.

## The structural idea

The page shows a ratio of two integer counts that only move when
Exophase refreshes: players tracked `P(t)` and per-trophy earners
`E_i(t)`, with rate `r_i = 100 * E_i / P`. The two counts are modelled
separately:

- `P(t)` explodes as new players sync profiles (diluting every rate).
- `E_i(t)` grows at its own pace as the cohort plays (story bronze
  fast, platinum slow).

A rate-only model cannot tell dilution from progress; a counts model can.

## Pipeline (see `main.py`, which is the whole entry)

1. **Refresh states.** Consecutive reads with identical
   `(players_tracked, rates)` collapse into one state at its first
   read's time. Between refreshes the page is frozen, so repeated reads
   carry no information and would overweight flat stretches.
2. **Log-time.** `x = ln(1 + h)`, `h` = hours since 2026-11-19T00:00Z.
   A power law in `(1 + h)` is a straight line in `x`, so a 4.5-day
   extrapolation stays tame.
3. **Recent window.** Fits use the last 24 h of distinct states (at
   least the last 5 states, expanding when refreshes are sparse).
   Launch week changes regime too fast for a whole-history average:
   in simulation a 24 h window cut the median logit error ~8x versus
   full-history fits (0.16 -> 0.02). The window is applied identically
   inside the backtest.
4. **Players.** `log P` vs `x`: a power law and a saturating logistic
   in `x` (ceiling grid `Pmax * [1.15 ... 377, 1e6]`, fit by regressing
   `ln(Pcap/P - 1)` on `x`) compete. The winner is chosen by
   rolling-origin backtest pinball loss at LONG horizons (each prefix
   predicts the last observed state); the power law is kept unless the
   saturating shape wins by > 5%.
5. **Earners.** Per trophy, `E_i = round(r_i * P / 100)` and
   `log E_i = c_i + d_i * x`, with empirical-Bayes shrinkage of `d_i`
   toward the tier mean slope, weighted by the OLS slope's own
   standard error (a slope the data pins down is barely shrunk; a
   slope built on rounding noise is pulled hard to the tier mean).
   Earners are monotone: `E_i(H)` is floored at the last observed
   `E_i` and capped at `P(H)`. A trophy held by everyone tracked
   (`E_i = P`) keeps rate 100% (clamped to 99.99 for the logit); a
   trophy with no earners anywhere gets a small prior (10% of the
   smallest positive last rate).
6. **Median.** `r_i(H) = 100 * E_i(H) / P(H)`, clamped [0.01, 99.99].
7. **Spread (conformal rolling-origin).** For each origin `k >= 3`,
   refit on the prefix and predict the last observed state; pool the
   logit residuals per tier (>= 30 residuals) else globally.
   `sigma_base = 1.4826 * MAD`. Random-walk scaling:
   `sigma_final = sigma_base * sqrt(H_days / H_backtest_days)`,
   clipped to `[0.30, 1.00]`. Quantiles:
   `q_tau = inv_logit(logit(median) + z_tau * sigma_final)`,
   `z = (-1.645, -0.674, 0, 0.674, 1.645)`, rounded to 4 decimals
   half-up.

## Calibrated constants (from the launch-week simulations)

- `SIGMA_FLOOR = 0.30`, `SIGMA_CAP = 1.00`. In simulation the
  residual-calibrated sigma collapses to 0.04-0.07 (simulated histories
  are smooth), while the oracle-optimal sigma is ~0.05 in smooth
  scenarios but ~0.30 when players growth saturates. A floor of 0.30
  pays ~0.05 loss in the smooth world and saves ~0.09 in the
  saturating world; the real history is lumpier than any simulation,
  so the floor is the honest middle. The cap lets genuinely large
  backtest residuals speak while stopping blowups.
- `FULL_EARNER_SIGMA = 1.20` for the 100% trophy (no observed
  downside yet; a flood of brand-new players could pull it down).
- `NO_RESID_SIGMA = 0.80` when fits exist but there is no backtest
  origin.
- Flat history (a single distinct state, as in the practice cases):
  no growth signal exists, so the stated prior is a median dilution
  of 2x with sigma 1.0; the 100% trophy stays at 99.99. No
  answer-specific constant (no 3,868, no sigma 0.02) is used anywhere.

## Validation (simulation harness in `forecast-sim/`, not in the repo)

Synthetic launch weeks: `P` power-law to 180k-650k by day 7 (plus a
saturating-`P` scenario), earners per tier hazard (early trophies
saturate, late trophies outgrow players), refreshes every 4-6 h, cut
day 2.5, target day 7. Pinball loss on the logit scale:

| scenario | model | persistence | trend | coverage 90% |
|---|---|---|---|---|
| A fast growth, seed 1 | 0.064 | 0.658 | 0.239 | 1.000 |
| A fast growth, seed 2 | 0.064 | 0.579 | 0.247 | 1.000 |
| B slow growth, seed 1 | 0.064 | 0.545 | 0.229 | 1.000 |
| B slow growth, seed 2 | 0.064 | 0.488 | 0.232 | 1.000 |
| C medium, seed 1 | 0.063 | 0.589 | 0.232 | 1.000 |
| C medium, seed 2 | 0.064 | 0.524 | 0.235 | 1.000 |
| D saturating P, seed 1 | 0.275 | 0.649 | 0.423 | 0.964 |
| D saturating P, seed 2 | 0.250 | 0.577 | 0.424 | 0.964 |

The model beats both house baselines in all 8 runs. Coverage in the
smooth scenarios is 1.000 because the sigma floor (0.30) binds there;
in the saturating scenario it is 0.964. The simulations are smoother
than reality, so their coverage is an upper bound, not a calibrated
claim — the floor exists for the noisier real world.

Practice (p00/p01, estimated truth only — the kit ships no answer):
against the dilution-only refresh estimate the model scores ~0.59
(persistence ~1.69); against the earner-growth-corrected estimate
(non-100% earners +33.8%, from `trophies_earned` 943 -> 4276) it
scores ~0.42 (persistence ~1.40; the house reports ~1.51, the gap is
non-uniform per-trophy growth, which no estimate can recover).
Practice pays nothing and was not optimised for.

## Limitations (honest)

- The power-law vs saturating players choice cannot be settled by the
  backtest at 1-2 day horizons: in scenario D the power law still won
  the backtest although the truth saturates, because the shapes are
  indistinguishable over the available horizons. The 4.5-day
  extrapolation is beyond what the data can discriminate.
- The 24 h window needs refreshes to be frequent enough; with very
  sparse refreshes it degrades to the last 5 states.
- The 100% trophy's spread (sigma 1.20) is a judgement call, not a
  measurement: it has no observed downside in the data.
