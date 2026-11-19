"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate.

Model: earners and tracked players are forecast separately, then combined.

  rate_i(target) = 100 * E_i(target) / N(target)

- N(t), the players Exophase tracks, is fitted as a power law in time since
  release: log N = a + b * log(1 + t_hours). Player counts grow fast at
  launch and the power law captures the slowdown better than a line.
- E_i(t), the earners of trophy i, is fitted the same way per trophy,
  log E_i = c_i + d_i * log(1 + t_hours), with each slope shrunk toward its
  tier's mean slope (trophies of a tier share how fast players reach them,
  so a trophy with a flat or noisy history borrows strength from its tier).
- The fits run on *distinct* reads only: Exophase refreshes in steps, so
  long stretches of byte-identical reads carry no new information and must
  not flatten the slopes.
- Stale-page fallback: when the whole history is a single distinct read
  (the launch-night practice cases), no slope can be fitted. The page is
  frozen while the real player base grows, so the next refresh dilutes every
  rate. The RULES state the one observed refresh: 638 players tracked at
  23:48 UTC, 3,868 at 04:30 UTC (x6.06 over ~4.7 h), every rate but the one
  at 100% lower. The fallback therefore assumes the tracked count keeps
  multiplying ~6x per 4.7 h of frozen time, while earners grow only
  part-way toward that (early trophies faster: new players earn them within
  hours; late trophies slower). Median rate = last rate * earner growth /
  player growth.
- Spread (logit scale): a random-walk term growing with the horizon, the
  pooled residual scatter of the fits, and a staleness term when the page
  has been frozen. Practice horizons need a wide band: the house baseline's
  narrow band (sigma 0.125 logit over 90 min) covered ~0% of the refreshed
  rates, so a frozen page gets sigma ~1.1 logit.

Deterministic, standard library only, no network. Reads every *.in file in
the input directory and writes one *.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The one refresh the RULES describe: tracked players 638 -> 3,868 over the
# frozen stretch from 2026-11-18T23:48Z to the 04:30Z read (~4.7 h).
REFRESH_MULTIPLY = 3868.0 / 638.0   # ~6.06x
REFRESH_SPAN_H = 4.7
# Share of the player-count growth that earners keep up with, per unit of
# a trophy's "earliness" (its current rate / 2%, capped at 1).
EARNER_KEEP_UP = 0.30
# Random-walk volatility of a rate's logit, per sqrt(day), as in the house
# baseline (which uses 0.5). 0.4 here: the sim backtest shows 0.3-0.4 beats
# 0.5 when the world is smooth, but the real page refreshes in jumps, so
# the band is not tightened to the simulation's smooth-world optimum.
SIGMA_RW = 0.4


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's JS rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t):
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def ols(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b, resid_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return a, b, resid


def distinct_reads(history):
    """Collapse consecutive identical reads (same players + same rates)."""
    out = []
    prev = None
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev:
            out.append(h)
            prev = key
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(hours_since_release(target))

    last = history[-1]
    last_rates = last["rates"]
    last_n = last["players_tracked"] or 1

    distinct = distinct_reads(history)
    n_distinct = len(distinct)

    # Recent window (last 48 h) for all fits: launch growth slows over
    # days, and a fit over the whole history would drag the early steep
    # slope into the forecast.
    cutoff = checkpoint - timedelta(hours=48)
    win = [h for h in distinct if parse_time(h["at"]) >= cutoff] or distinct
    if len(win) < 2:
        win = distinct

    # ---- tracked players N(t): power law in log(1 + hours since release).
    if n_distinct >= 2:
        xs = [math.log1p(hours_since_release(parse_time(h["at"])))
              for h in win]
        ys = [math.log(max(1, h["players_tracked"] or 1)) for h in win]
        a_n, b_n, _ = ols(xs, ys)
        b_n = min(2.5, max(0.0, b_n))
        n_target = math.exp(a_n + b_n * x_target)
        n_target = min(max(n_target, last_n * 0.5), last_n * 50.0)
    else:
        n_target = None  # stale-page fallback below

    # ---- per-trophy earners E_i(t), slope shrunk toward the tier mean.
    # First pass: raw per-trophy slopes, to form tier means.
    raw_slopes = {}
    tier_slopes = {}
    if n_distinct >= 2:
        xs = [math.log1p(hours_since_release(parse_time(h["at"])))
              for h in win]
        for i, tr in enumerate(trophies):
            ys = []
            ok = True
            for h in win:
                r = h["rates"][i]
                if r is None:
                    ok = False
                    break
                e = max(0.5, (r / 100.0) * (h["players_tracked"] or 1))
                ys.append(math.log(e))
            if not ok or len(set(ys)) == 1 and len(ys) < 3:
                continue
            _, b, _ = ols(xs, ys)
            raw_slopes[i] = min(4.0, max(-0.5, b))
            tier_slopes.setdefault(tr["tier"], []).append(raw_slopes[i])
    tier_mean = {t: sum(v) / len(v) for t, v in tier_slopes.items()}
    global_mean = (sum(raw_slopes.values()) / len(raw_slopes)
                   if raw_slopes else 0.5)

    # Pooled residual scatter of the log-earner fits (spread calibration).
    resid_pool = []
    if n_distinct >= 3:
        xs = [math.log1p(hours_since_release(parse_time(h["at"])))
              for h in win]
        for i, tr in enumerate(trophies):
            ys = []
            ok = True
            for h in win:
                r = h["rates"][i]
                if r is None:
                    ok = False
                    break
                ys.append(math.log(max(0.5, (r / 100.0)
                                       * (h["players_tracked"] or 1))))
            if ok and len(ys) >= 3:
                _, _, resid = ols(xs, ys)
                resid_pool.append(resid)
    sigma_fit = sorted(resid_pool)[len(resid_pool) // 2] if resid_pool else 0.30
    sigma_fit = min(0.8, max(0.15, sigma_fit))

    # How long the page has sat unchanged at the checkpoint (staleness).
    flat_hours = 0.0
    if n_distinct >= 1:
        last_change = parse_time(distinct[-1]["at"])
        flat_hours = max(0.0, (checkpoint - last_change).total_seconds() / 3600.0)

    rows = []
    for i, tr in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            # Trophy missing from the last read: fall back to its last rate.
            for h in reversed(history):
                if h["rates"][i] is not None:
                    rate = h["rates"][i]
                    break
        if rate is None:
            rate = 0.01

        if rate >= 100.0:
            # Earned by everyone tracked, in every read so far: it stays
            # ~100% (the one rate the observed refresh did not lower).
            median = 99.99
            sigma = max(0.35, SIGMA_RW * math.sqrt(horizon_days))
        elif n_target is None:
            # ---- stale-page fallback (frozen launch-night page).
            frozen_h = max(0.0, (target - parse_time(history[0]["at"]))
                           .total_seconds() / 3600.0)
            mult = REFRESH_MULTIPLY ** (frozen_h / REFRESH_SPAN_H)
            mult = min(mult, 12.0)
            earliness = min(1.0, rate / 2.0)
            earner_growth = 1.0 + (mult - 1.0) * earliness * EARNER_KEEP_UP
            median = clamp(rate * earner_growth / mult)
            sigma = 1.10
        else:
            # ---- fitted forecast: E_i(target) / N(target).
            e_last = max(0.5, (rate / 100.0) * last_n)
            x_last = math.log1p(hours_since_release(parse_time(last["at"])))
            if i in raw_slopes and len(win) >= 3:
                tier = tier_mean.get(tr["tier"], global_mean)
                w = len(win) / (len(win) + 3.0)
                slope = w * raw_slopes[i] + (1.0 - w) * tier
            elif i in raw_slopes:
                slope = raw_slopes[i]
            else:
                slope = tier_mean.get(tr["tier"], global_mean)
            slope = min(4.0, max(-0.5, slope))
            e_target = e_last * math.exp(slope * (x_target - x_last))
            median = clamp(100.0 * e_target / n_target)
            sigma = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                              + sigma_fit ** 2
                              + (0.30 * min(1.0, flat_hours / 6.0)) ** 2)
            sigma = min(2.0, max(0.35, sigma))

        lg = logit(median)
        q = [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": tr["id"], "q": q})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels", LEVELS) != LEVELS:
            raise ValueError(f"{path.name} asks for other levels")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
