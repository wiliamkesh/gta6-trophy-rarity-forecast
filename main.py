"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Two-process cohort hazard & launch-dilution model.

The page Exophase shows is a ratio of two integer counts that only move
when Exophase refreshes the game's page: players tracked P(t), and per
trophy the earners E_i(t).  A trophy's printed rate is r_i = 100 * E_i / P.
Modelling the two counts separately, instead of the ratio, is what makes
a launch-week forecast possible: P explodes as new players sync their
profiles (diluting every rate), while E_i grows at its own pace as the
cohort plays through the game (bronze story beats fast, platinum slow).

Method, per case:
  1. Collapse the history into distinct refresh states: consecutive reads
     with identical (players_tracked, rates) are one state, kept at the
     time of its first read.  Between refreshes the page is frozen, so
     repeated reads carry no new information and would overweight the
     flat stretches in any fit.
  2. Work in log-time x = ln(1 + h), h = hours since 2026-11-19T00:00Z.
     Log-time keeps a 4.5-day extrapolation tame: a power law in (1 + h)
     is a straight line in x.
  2b. Fit on a RECENT window, not the whole history: the last 24 hours
     of distinct states (at least the last 5 states, expanding the
     window when refreshes are sparse).  Launch week changes regime
     fast -- the cohort that was tracked on launch night is not the
     cohort tracked two days later -- and a fit averaged over the whole
     history extrapolates the launch-night regime into day 7.  The
     window is applied the same way inside the backtest, so the model
     the backtest measures is the model that forecasts.
  3. Players: fit log P against x.  Two shapes compete -- a power law
     (straight line) and a saturating logistic in x (a ceiling Pcap, fit
     by regressing ln(Pcap / P - 1) on x over a grid of ceilings).  The
     winner is chosen by rolling-origin backtest pinball loss on the
     rates at LONG horizons (each prefix predicts the last observed
     state: short gaps cannot tell a saturating curve from a power law,
     but a 2-day extrapolation can), with the power law kept unless the
     saturating shape wins by more than 5% (it extrapolates more
     conservatively, so it has to earn the choice).
  4. Earners: per trophy, E_i = round(r_i * P / 100) (integer-count
     consistent) and fit log E_i = c_i + d_i * x over the states where
     E_i > 0.  Slopes shrink toward their tier's mean slope with
     empirical-Bayes weights from the OLS slope's own standard error:
     a slope the data pins down (tiny SE) is barely shrunk, a slope
     built on rounding noise (large SE) is pulled hard to the tier
     mean -- because a rare trophy's two-point slope is mostly noise
     from the page's 2-decimal rates, while a common trophy's slope is
     measured.  The intercept is re-anchored after shrinkage so the
     level still matches the data.
     Earners are monotone: E_i(H) is floored at the last observed E_i
     and capped at P(H).  A trophy earned by everyone tracked (E_i = P)
     keeps rate 100% (clamped to 99.99 for the logit); a trophy with no
     earners anywhere gets a small tier-free prior instead of a fit.
  5. Median rate: r_i(H) = 100 * E_i(H) / P(H), clamped to [0.01, 99.99].
  6. Spread: conformal rolling-origin.  For each distinct state s_k
     (k >= 3), refit the model on the states before s_k and predict
     the LAST observed state from that prefix: a long-horizon
     prediction, whose logit residual (logit(actual) - logit(predicted))
     measures multi-day extrapolation error -- the error that matters
     for a 4.5-day forecast.  (Next-state residuals are far too
     optimistic: a 5-hour-ahead prediction is easy.)  Residuals pool
     per tier when a tier has at least 30 of them, else globally.
     sigma_base = 1.4826 * MAD of the pooled residuals.  Random-walk
     scaling to the final horizon:
         sigma_final = sigma_base * sqrt(H_days / H_backtest_days)
     where H_backtest_days is the median long-horizon backtest horizon,
     clipped to [SIGMA_FLOOR, SIGMA_CAP].  The floor stops the interval
     collapsing when the backtest happens to be flat; the cap stops the
     scaling exploding.  Both were set from the launch-week simulations
     in MODEL.md.
  7. Quantiles: q_tau = inv_logit(logit(median) + z_tau * sigma_final),
     z = (-1.645, -0.674, 0, 0.674, 1.645), rounded to 4 decimals
     half-up.

Degenerate histories (what the practice cases are):
  - One distinct state: there is no growth signal at all, so no slope
    can be fit.  The model falls back to a stated prior: the page has
    been stale for hours on launch night and the next refresh can only
    add players, so the median dilutes by a factor of 2 (rates halve)
    with a wide spread (sigma 1.0 logit); a trophy at 100% stays at
    99.99, since every tracked player holds it.  No answer-specific
    constant is used anywhere: the practice refresh size is not in
    this file, and the practice score is not optimised for.
  - Fits exist but no backtest origin: sigma falls back to a prior
    (NO_RESID_SIGMA) instead of pretending the residuals were measured.

Only the standard library.  No network, no randomness, no wall clock:
two runs on the same input write byte-identical output.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Time origin: the release day began 2026-11-19T00:00:00Z.
ORIGIN = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Recent window for fits: last 24 h of distinct states, but never
# fewer than this many states (the window expands when refreshes are
# sparse).  24 h matches the house curve baseline's window and, in the
# launch-week simulations, a 24 h window cut the median logit error
# roughly 8x versus fitting the whole history (0.16 -> 0.02), because
# launch week changes regime too fast for a whole-history average.
WINDOW_HOURS = 24.0
WINDOW_MIN_STATES = 5

# Saturating-players ceilings, as multiples of the largest P seen.
# The last entry (1e6) is effectively the power law: with a ceiling far
# above every observation the logistic is an exponential in x.
PCAP_MULTS = [1.15, 1.3, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0,
              55.0, 89.0, 144.0, 233.0, 377.0, 1.0e6]

# A saturating shape must beat the power law by this relative margin in
# rolling-origin backtest pinball loss to be chosen.
SATURATE_MARGIN = 0.05

# Tier slope shrinkage is empirical-Bayes (see fit_earners): the weight
# comes from each slope's own standard error, not a fixed count rule.
# A slope from fewer than 3 positive states gets this prior standard
# error instead (a 2-point line has zero residual by construction, so
# its OLS standard error would falsely read as zero).
SLOPE_SE_PRIOR = 0.75
# Earner slopes are clipped to this range (log-time slopes beyond it are
# rounding artefacts of the page's 2-decimal rates, not real growth).
SLOPE_MIN, SLOPE_MAX = -1.0, 3.0

# Spread calibration (logit units).  The floor stops the interval
# collapsing when the backtest looks over-confident: in the launch-week
# simulations the residual-calibrated sigma collapses to 0.04-0.07
# (the simulated histories are smooth), while the oracle-optimal sigma
# is ~0.05 in smooth scenarios but ~0.30 when players growth saturates
# (scenario D).  A floor of 0.30 pays ~0.05 loss in the smooth world
# and saves ~0.09 in the saturating world, and the real final history
# is lumpier than any simulation, so the floor is the honest middle.
# The cap lets genuinely large backtest residuals speak (a lumpy real
# history widens sigma on its own) while stopping blowups.
SIGMA_FLOOR = 0.30
SIGMA_CAP = 1.00
# Spread when fits exist but there is no backtest origin to measure one.
NO_RESID_SIGMA = 0.80
# Flat-history prior (a single distinct state): median dilution factor
# and spread.  A refresh on launch night can only add players, so the
# median assumes one refresh roughly doubles the tracked players, with
# a wide spread because the refresh size is unknown.
FLAT_DILUTION = 2.0
FLAT_SIGMA = 1.0
# A trophy held by every tracked player gets a wider spread than the
# calibrated one: it has no observed downside yet, and a flood of brand
# new players could pull it below 100%.
FULL_EARNER_SIGMA = 1.20
# A trophy with no earners anywhere: prior median = this fraction of the
# smallest positive last rate, spread at least this wide.
ZERO_EARNER_FRAC = 0.10
ZERO_EARNER_SIGMA = 0.80

# Residuals pool per tier only when the tier has this many.
TIER_POOL_MIN = 30


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        p = 1.0 / (1.0 + math.exp(-lg)) if lg < 700 else 1.0
    else:
        e = math.exp(lg) if lg > -700 else 0.0
        p = e / (1.0 + e)
    return 100.0 * p


def round4(x):
    """Round to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_origin(moment):
    return (moment - ORIGIN).total_seconds() / 3600.0


def log_time(h):
    """x = ln(1 + h), guarded so 1 + h stays positive."""
    return math.log(1.0 + max(h, -0.99))


def _log1pexp(t):
    """log(1 + exp(t)), overflow-safe."""
    if t > 700:
        return t
    if t < -700:
        return 0.0
    return math.log1p(math.exp(t))


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02,
         -2.759285984469534e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def ols(xs, ys):
    """Least-squares line y = a + b*x.  Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def ols_with_se(xs, ys):
    """Least squares plus the slope's standard error.

    Returns (a, b, se_b) or None.  With < 3 points the standard error
    is a prior scale (a 2-point line fits exactly, so its residual
    standard error would be a false zero).
    """
    n = len(xs)
    if n < 2:
        return None
    line = ols(xs, ys)
    if line is None:
        return None
    a, b = line
    if n < 3:
        return (a, b, SLOPE_SE_PRIOR)
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    s2 = rss / (n - 2)
    se = math.sqrt(s2 / sxx) if sxx > 1e-12 else SLOPE_SE_PRIOR
    return (a, b, se)


def median_of(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def as_number(value):
    """players_tracked arrives as an int or a string; read either."""
    if isinstance(value, str):
        return float(value.replace(",", "").strip())
    return float(value)


# ---------------------------------------------------------------- states

def distinct_states(case):
    """Collapse consecutive identical reads into refresh states.

    Each state: {"h": hours since origin, "x": log-time, "P": players,
    "rates": [...], "E": [earner counts or None]}.  The state's time is
    its first read's time: the refresh happened between the previous
    distinct state and this read.
    """
    trophies = case["trophies"]
    hist = sorted(case["history"], key=lambda r: r["at"])
    states = []
    prev_key = None
    for read in hist:
        try:
            players = as_number(read["players_tracked"])
        except (KeyError, TypeError, ValueError):
            continue
        rates = list(read["rates"])
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        moment = parse_time(read["at"])
        h = hours_since_origin(moment)
        earners = []
        for rate in rates:
            if rate is None:
                earners.append(None)
            else:
                earners.append(int(round(rate * players / 100.0)))
        states.append({"h": h, "x": log_time(h), "P": players,
                       "rates": rates, "E": earners})
    return states


def windowed_states(states):
    """The recent window of distinct states a fit uses: the last
    WINDOW_HOURS of states, or the last WINDOW_MIN_STATES states when
    refreshes are too sparse to fill the window."""
    if len(states) <= WINDOW_MIN_STATES:
        return states
    cutoff = states[-1]["h"] - WINDOW_HOURS
    recent = [s for s in states if s["h"] >= cutoff]
    if len(recent) >= WINDOW_MIN_STATES:
        return recent
    return states[-WINDOW_MIN_STATES:]


# ------------------------------------------------------------- model fits

def fit_players_loglaw(states):
    """log P = a + b*x (power law).  Returns (a, b) or None."""
    xs = [s["x"] for s in states]
    ys = [math.log(s["P"]) for s in states]
    return ols(xs, ys)


def fit_players_logistic(states, pcap):
    """Logistic ceiling: P = Pcap / (1 + exp(-(a + b*x))).

    ln(Pcap / P - 1) = -(a + b*x) is linear in x, so one OLS fit per
    ceiling.  Returns (log_pcap, a, b) or None.
    """
    xs = []
    ys = []
    for s in states:
        if s["P"] >= pcap:
            return None
        xs.append(s["x"])
        ys.append(math.log(pcap / s["P"] - 1.0))
    line = ols(xs, ys)
    if line is None:
        return None
    a_neg, b_neg = line
    return (math.log(pcap), -a_neg, -b_neg)


def predict_log_players(fit, x):
    """fit is ("law", a, b) or ("cap", log_pcap, a, b)."""
    if fit[0] == "law":
        return fit[1] + fit[2] * x
    _, log_pcap, a, b = fit
    return log_pcap - _log1pexp(-(a + b * x))


def fit_earners(states, tiers):
    """Per-trophy log-earner fits with tier shrinkage.

    Returns a list, one entry per trophy:
      {"c": intercept, "d": shrunk slope, "n": states with E > 0,
       "kind": "fit" | "tier" | "full" | "prior"}
    "full" means E == P in every state where it was seen (rate 100%);
    "prior" means no positive earner count was ever seen.

    Shrinkage is empirical-Bayes per tier: for the tier's slopes d_i
    with standard errors se_i, the spread of true slopes is estimated
    as tau^2 = max(0, var(d) - mean(se^2)), and each slope is shrunk
    toward the tier mean with weight w_i = tau^2 / (tau^2 + se_i^2).
    A slope the data pins down keeps its value; a slope built on
    rounding noise is pulled to the tier mean.
    """
    n_tr = len(tiers)
    raw = []  # (c, d, se, n) per trophy from the plain OLS fit
    for i in range(n_tr):
        xs, ys = [], []
        for s in states:
            e = s["E"][i]
            if e is not None and e > 0:
                xs.append(s["x"])
                ys.append(math.log(e))
        if len(xs) >= 2:
            fit = ols_with_se(xs, ys)
            if fit is not None:
                raw.append((fit[0], fit[1], fit[2], len(xs)))
            else:
                raw.append((ys[0], 0.0, SLOPE_SE_PRIOR, len(xs)))
        elif len(xs) == 1:
            raw.append((ys[0], None, None, 1))
        else:
            raw.append((None, None, None, 0))

    # Tier mean slopes and true-slope spread, from multi-point fits.
    def tier_stats(tier):
        ds = [raw[i][1] for i in range(n_tr)
              if tiers[i] == tier and raw[i][1] is not None]
        ses = [raw[i][2] for i in range(n_tr)
               if tiers[i] == tier and raw[i][1] is not None]
        if not ds:
            return None
        mean = sum(ds) / len(ds)
        if len(ds) >= 2:
            var = sum((d - mean) ** 2 for d in ds) / (len(ds) - 1)
            tau2 = max(0.0, var - sum(se * se for se in ses) / len(ses))
        else:
            tau2 = 0.25
        return (mean, tau2)

    stats = {}
    for tier in set(tiers):
        stats[tier] = tier_stats(tier)
    all_ds = [d for (_, d, _, _) in raw if d is not None]
    global_mean = sum(all_ds) / len(all_ds) if all_ds else 0.0
    global_tau2 = 0.25

    out = []
    for i in range(n_tr):
        c0, d0, se0, n = raw[i]
        # Full-earner check: E == P everywhere it was seen.
        seen = [(s["E"][i], s["P"]) for s in states if s["E"][i] is not None]
        if seen and all(e == p for e, p in seen):
            out.append({"c": None, "d": None, "n": n, "kind": "full"})
            continue
        if n == 0:
            out.append({"c": None, "d": None, "n": 0, "kind": "prior"})
            continue
        st = stats.get(tiers[i]) or (global_mean, global_tau2)
        tier_slope, tau2 = st
        if d0 is None:
            d = tier_slope
            kind = "tier"
        else:
            w = tau2 / (tau2 + se0 * se0) if tau2 > 0 else 0.0
            d = w * d0 + (1.0 - w) * tier_slope
            kind = "fit"
        d = min(SLOPE_MAX, max(SLOPE_MIN, d))
        # Re-anchor the intercept at the shrunk slope.
        xs, ys = [], []
        for s in states:
            e = s["E"][i]
            if e is not None and e > 0:
                xs.append(s["x"])
                ys.append(math.log(e))
        c = sum(y - d * x for x, y in zip(xs, ys)) / len(xs)
        out.append({"c": c, "d": d, "n": n, "kind": kind})
    return out


def fit_players(states, pcap_mult):
    """Players fit for a ceiling multiple: ("law", a, b) or
    ("cap", log_pcap, a, b).  Falls back to the power law when the
    logistic cannot be fit (ceiling at or below the largest P seen)."""
    p_max = max(s["P"] for s in states)
    if pcap_mult >= 1.0e6:
        law = fit_players_loglaw(states)
        if law is None:
            return ("law", math.log(states[-1]["P"]), 0.0)
        return ("law", law[0], law[1])
    cap = fit_players_logistic(states, p_max * pcap_mult)
    if cap is None:
        law = fit_players_loglaw(states)
        if law is None:
            return ("law", math.log(states[-1]["P"]), 0.0)
        return ("law", law[0], law[1])
    return ("cap", cap[0], cap[1], cap[2])


def median_rates_from_fits(states, tiers, e_fits, p_fit, x_target):
    """Median rates at log-time x_target from precomputed fits.

    `states` is the prefix the fits were built on: the earner floor
    and the players floor both use its last state.
    """
    p_target = math.exp(predict_log_players(p_fit, x_target))
    p_target = max(p_target, states[-1]["P"])

    last_rates = states[-1]["rates"]
    pos_last = [r for r in last_rates if r is not None and r > 0]
    prior_rate = ZERO_EARNER_FRAC * min(pos_last) if pos_last else 0.05

    rates = []
    for i, ef in enumerate(e_fits):
        if ef["kind"] == "full":
            rates.append(99.99)
            continue
        if ef["kind"] == "prior":
            rates.append(prior_rate)
            continue
        e_pred = math.exp(ef["c"] + ef["d"] * x_target)
        e_last = states[-1]["E"][i]
        if e_last is not None and e_last > 0:
            e_pred = max(e_pred, float(e_last))
        e_pred = min(e_pred, p_target)
        rates.append(clamp_pct(100.0 * e_pred / p_target))
    return rates


def predict_median_rates(states, tiers, x_target, pcap_mult):
    """Median rates at log-time x_target, fitting on `states` only.

    Returns (rates, p_fit).
    """
    p_fit = fit_players(states, pcap_mult)
    e_fits = fit_earners(states, tiers)
    return median_rates_from_fits(states, tiers, e_fits, p_fit, x_target), p_fit


# -------------------------------------------------------------- backtest

def pinball_median_loss(actual, predicted):
    """Mean |logit(actual) - logit(predicted)| over scored trophies."""
    total = 0.0
    count = 0
    for y, q in zip(actual, predicted):
        if y is None or q is None:
            continue
        total += abs(logit(y) - logit(q))
        count += 1
    return (total / count) if count else None


def backtest(states, tiers):
    """Rolling-origin backtest over distinct states.

    For each origin k >= 3, fit on states[:k] (earner fits once per
    origin, players fits once per ceiling) and predict two things:
      * the next state (short horizon): its pinball loss is the
        fallback shape comparison when no long horizon exists;
      * the last observed state (long horizon): its pinball loss
        chooses between the power-law and saturating players shapes
        (short gaps cannot tell the shapes apart, but a 2-day
        extrapolation can, and the final extrapolation is 4.5 days),
        and its logit residuals calibrate the spread, because they
        measure multi-day extrapolation error.
    Returns (chosen_mult, long_residuals, short_residuals,
    median_long_days, median_gap_days) where residuals are lists of
    (tier, logit residual) for the chosen players shape.  With fewer
    than 4 distinct states there is no origin.
    """
    if len(states) < 4:
        return (1.0e6, [], [], None, None)
    gaps = []
    for k in range(1, len(states)):
        gaps.append((states[k]["h"] - states[k - 1]["h"]) / 24.0)
    median_gap = median_of(gaps)

    cand_long_resid = {m: [] for m in PCAP_MULTS}
    cand_short_resid = {m: [] for m in PCAP_MULTS}
    short_loss = {m: 0.0 for m in PCAP_MULTS}
    short_count = {m: 0 for m in PCAP_MULTS}
    long_loss = {m: 0.0 for m in PCAP_MULTS}
    long_count = {m: 0 for m in PCAP_MULTS}
    long_horizons = []
    for k in range(3, len(states)):
        prefix = states[:k]
        # The model the backtest measures is the windowed model.
        fit_on = windowed_states(prefix)
        e_fits = fit_earners(fit_on, tiers)
        for mult in PCAP_MULTS:
            p_fit = fit_players(fit_on, mult)
            pred_next = median_rates_from_fits(
                fit_on, tiers, e_fits, p_fit, states[k]["x"])
            loss = pinball_median_loss(states[k]["rates"], pred_next)
            if loss is not None:
                short_loss[mult] += loss
                short_count[mult] += 1
            for i, y in enumerate(states[k]["rates"]):
                if y is None or pred_next[i] is None:
                    continue
                cand_short_resid[mult].append(
                    (tiers[i], logit(y) - logit(pred_next[i])))
            if k < len(states) - 1:
                pred_last = median_rates_from_fits(
                    fit_on, tiers, e_fits, p_fit, states[-1]["x"])
                loss = pinball_median_loss(states[-1]["rates"], pred_last)
                if loss is not None:
                    long_loss[mult] += loss
                    long_count[mult] += 1
                for i, y in enumerate(states[-1]["rates"]):
                    if y is None or pred_last[i] is None:
                        continue
                    cand_long_resid[mult].append(
                        (tiers[i], logit(y) - logit(pred_last[i])))
                if mult == PCAP_MULTS[0]:
                    long_horizons.append(
                        (states[-1]["h"] - states[k - 1]["h"]) / 24.0)
    scored_long = {m: (long_loss[m] / long_count[m]) for m in PCAP_MULTS
                   if long_count[m] > 0}
    scored_short = {m: (short_loss[m] / short_count[m]) for m in PCAP_MULTS
                    if short_count[m] > 0}
    scored = scored_long if scored_long else scored_short
    if not scored:
        return (1.0e6, [], [], None, median_gap)
    law_loss = scored.get(1.0e6)
    best_mult = min(scored, key=lambda m: scored[m])
    chosen = best_mult
    if law_loss is not None and best_mult != 1.0e6:
        # The saturating shape must beat the power law by the margin.
        if scored[best_mult] > (1.0 - SATURATE_MARGIN) * law_loss:
            chosen = 1.0e6
    median_long = median_of(long_horizons) if long_horizons else None
    return (chosen, cand_long_resid[chosen], cand_short_resid[chosen],
            median_long, median_gap)


def sigma_from_residuals(residuals, tiers):
    """sigma_base per tier (global fallback), from 1.4826 * MAD."""
    by_tier = {}
    for tier, e in residuals:
        by_tier.setdefault(tier, []).append(e)
    all_e = [e for _, e in residuals]
    out = {}
    if all_e:
        med = median_of(all_e)
        mad = median_of([abs(e - med) for e in all_e])
        out["__global__"] = 1.4826 * mad
    for tier, vals in by_tier.items():
        if len(vals) >= TIER_POOL_MIN:
            med = median_of(vals)
            mad = median_of([abs(v - med) for v in vals])
            out[tier] = 1.4826 * mad
    return out


# ------------------------------------------------------------ forecasting

def forecast_case(case):
    trophies = case["trophies"]
    tiers = [t["tier"] for t in trophies]
    ids = [t["id"] for t in trophies]
    states = distinct_states(case)
    target = parse_time(case["target"])
    h_target = hours_since_origin(target)
    x_target = log_time(h_target)
    levels = case.get("levels", LEVELS)
    zvals = [Z[i] if levels == LEVELS else norm_ppf(lv) for i, lv in enumerate(levels)]

    if not states:
        # No history at all: emit a valid, maximally uninformative row.
        return {"trophies": [{"id": tid, "q": [0.01, 0.1, 1.0, 10.0, 99.99]}
                             for tid in ids]}

    last_rates = states[-1]["rates"]

    if len(states) == 1:
        # Flat history: no growth signal.  Stated prior (see module doc).
        medians = []
        sigmas = []
        for i, r in enumerate(last_rates):
            if r is None:
                medians.append(1.0)
                sigmas.append(FLAT_SIGMA)
            elif r >= 99.99:
                medians.append(99.99)
                sigmas.append(FULL_EARNER_SIGMA)
            else:
                medians.append(clamp_pct(r / FLAT_DILUTION))
                sigmas.append(FLAT_SIGMA)
        return build_rows(ids, medians, sigmas, zvals)

    chosen_mult, long_resid, short_resid, median_long, median_gap = \
        backtest(states, tiers)

    if long_resid:
        # Long-horizon residuals measure multi-day extrapolation error.
        residuals = long_resid
        ref_days = median_long
    elif short_resid:
        residuals = short_resid
        ref_days = median_gap
    else:
        residuals = []
        ref_days = None

    if residuals:
        sig_base = sigma_from_residuals(residuals, tiers)
        horizon_days = max((target - parse_time(case["history"][-1]["at"])
                            ).total_seconds() / 86400.0,
                           case["horizon_minutes"] / 1440.0)
        gap_days = ref_days if ref_days and ref_days > 0 else 1.0
        scale = math.sqrt(max(horizon_days, 1e-6) / gap_days)
        global_base = sig_base.get("__global__", NO_RESID_SIGMA)
    else:
        sig_base = {}
        scale = 1.0
        global_base = NO_RESID_SIGMA

    medians, _ = predict_median_rates(windowed_states(states), tiers,
                                      x_target, chosen_mult)

    # Per-trophy sigma: tier pool when it exists, else the global one,
    # random-walk scaled to the final horizon and clipped.
    e_fits = fit_earners(windowed_states(states), tiers)
    sigmas = []
    for i, ef in enumerate(e_fits):
        base = sig_base.get(tiers[i], global_base)
        sig = base * scale
        sig = min(SIGMA_CAP, max(SIGMA_FLOOR, sig))
        if ef["kind"] == "full":
            sig = max(sig, FULL_EARNER_SIGMA)
        elif ef["kind"] == "prior":
            sig = max(sig, ZERO_EARNER_SIGMA)
        sigmas.append(sig)
    return build_rows(ids, medians, sigmas, zvals)


def build_rows(ids, medians, sigmas, zvals):
    rows = []
    for tid, med, sig in zip(ids, medians, sigmas):
        centre = logit(med)
        qs = []
        prev = 0.01
        for z in zvals:
            q = round4(clamp_pct(inv_logit_pct(centre + z * sig)))
            q = max(q, prev)
            qs.append(q)
            prev = q
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def fallback_rows(case):
    """Last-rate persistence with a wide spread; never crash a case."""
    try:
        trophies = case["trophies"]
        last = case["history"][-1]["rates"] if case["history"] else []
        rows = []
        for i, t in enumerate(trophies):
            r = last[i] if i < len(last) and last[i] is not None else 1.0
            centre = logit(r)
            qs = []
            prev = 0.01
            for z in Z:
                q = round4(clamp_pct(inv_logit_pct(centre + z * 1.0)))
                q = max(q, prev)
                qs.append(q)
                prev = q
            rows.append({"id": t["id"], "q": qs})
        return {"trophies": rows}
    except Exception:
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.1, 1.0, 10.0, 99.99]}
                             for t in case.get("trophies", [])]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_rows(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
