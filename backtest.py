"""Backtest for gta6-trophy-rarity-forecast-01 main.py (simulation-based).

The kit publishes no practice answers, so a direct backtest is impossible.
Two honest substitutes:

1) SIMULATION: a synthetic launch week. Tracked players N(t) grow as a
   power law; each trophy's earners follow a progression curve in log-time
   with tier-dependent speed; Exophase refreshes in steps (page frozen
   between refreshes, reads sample the last refresh). Cases are cut at
   practice-like checkpoints (03:00/03:30, target +90 min) and at a
   final-like checkpoint (Nov 21 11:30 UTC, target Nov 26 00:00 UTC).
   True answers are the underlying rates at the target. Our program is
   scored against the starter (persistence) baseline and the house curve
   baseline (logit rate vs log(1+hours), last 24 h) re-implemented here.

2) PRACTICE SENSITIVITY: assume the practice answers equal last rate x 0.22
   for every trophy but the 100% one. The 0.22 comes from the RULES: the
   house baseline (median = last rate, near-zero spread) scores ~1.51 mean
   pinball on the practice, and with a near-degenerate forecast the pinball
   loss ~= |logit(y) - logit(median)|, so the mean answer is ~last x e^-1.51
   ~= last x 0.22. (Also consistent with the stated typical miss 0.68 pp
   and with dilution 638 -> 3,868 players partly offset by earner growth.)
   This is a scenario, not a measurement; the simulation is the real test.
"""

import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as prog

UTC = timezone.utc
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=UTC)
KIT = "/home/hatch/workspace/makefun/kits/gta6-trophy-rarity-forecast-01/" \
      "gta6-trophy-rarity-forecast-01-kit/input"

Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]


def pinball(y, q, tau):
    def logit(p):
        p = min(99.99, max(0.01, p)) / 100.0
        return math.log(p / (1.0 - p))
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(forecast, answers):
    tot, n = 0.0, 0
    for row, y in zip(forecast["trophies"], answers):
        for tau, q in zip(prog.LEVELS, row["q"]):
            tot += pinball(y, q, tau)
            n += 1
    return tot / n


def baseline_persistence(case):
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        lg = prog.logit(last["rates"][i] if last["rates"][i] is not None
                        else 0.01)
        rows.append({"id": tr["id"], "q": [
            prog.round4(prog.clamp(prog.inv_logit(lg + z * spread)))
            for z in Z]})
    return {"trophies": rows}


def baseline_curve(case):
    """House curve baseline: logit(rate) ~ log(1+hours) over last 24 h."""
    hist = case["history"]
    cutoff = prog.parse_time(case["at"]) - timedelta(hours=24)
    use = [h for h in hist if prog.parse_time(h["at"]) >= cutoff] or hist
    xs = [math.log1p(prog.hours_since_release(prog.parse_time(h["at"])))
          for h in use]
    x_t = math.log1p(prog.hours_since_release(prog.parse_time(case["target"])))
    spread_rw = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        ys = [prog.logit(h["rates"][i]) for h in use if h["rates"][i] is not None]
        if len(ys) < 3:
            lg = prog.logit(use[-1]["rates"][i] or 0.01)
            sig = spread_rw
        else:
            a, b, resid = prog.ols(xs, ys)
            lg = a + b * x_t
            sig = math.sqrt(spread_rw ** 2 + max(0.0, resid) ** 2)
        rows.append({"id": tr["id"], "q": [
            prog.round4(prog.clamp(prog.inv_logit(lg + z * sig))) for z in Z]})
    return {"trophies": rows}


# ---------- synthetic world ----------
# Calibrated to the observed launch-night state: the first read's rates are
# the kit's own p00 last rates (reviewers/early players, ~8 h effective
# playtime), tracked players 638 -> 3,868 at the 04:30 refresh. Earners then
# grow along a log-time progression curve; N(t) follows a power law.
def simulate(seed=7):
    rng = random.Random(seed)
    base = json.load(open(f"{KIT}/p00.in"))
    trophies = base["trophies"]
    last_rates = base["history"][-1]["rates"]
    P0 = 8.0  # effective playtime (h) of the first tracked cohort

    def s_at(tau, play_h):
        return 1.0 - math.exp(-((max(play_h, 0.0) + 0.5) / tau) ** 0.8)

    truth = []
    for tr, r0 in zip(trophies, last_rates):
        if tr["id"] == 2:
            truth.append((100.0, 0.5))
            continue
        tau = {"platinum": 400.0, "gold": 160.0, "silver": 90.0,
               "bronze": 55.0}[tr["tier"]] * math.exp(rng.gauss(0, 0.5))
        truth.append((r0, tau))
    # N(t): power law that slows after 12 h (exponent 1.3 -> 0.6):
    # 638 at t=-0.17h, ~3,900 at 4.5h, ~28k at the deadline, ~50k at target.
    def n_true(t_h):
        t = t_h + 0.17
        base = (1.0 + min(t, 12.17) / 1.5) ** 1.3
        if t > 12.17:
            base *= ((1.0 + t / 1.5) / (1.0 + 12.17 / 1.5)) ** 0.6
        return 638.0 * base
    def rate_true(i, t_h):
        r0, tau = truth[i]
        if r0 >= 100.0:
            return 100.0
        play = P0 + (t_h + 0.17)  # effective playtime grows with clock
        earners = (r0 / 100.0) * 638.0 * (s_at(tau, play) / s_at(tau, P0))
        return min(100.0, 100.0 * earners / n_true(t_h))
    return trophies, truth, n_true, rate_true


def build_case(trophies, n_true, rate_true, checkpoint, horizon_min,
               refresh_every_min, case_id):
    """Reads every 30 min (early) / 10 min, page frozen between refreshes."""
    target = checkpoint + timedelta(minutes=horizon_min)
    # refresh times: every refresh_every_min from release-10min
    refreshes = []
    t = RELEASE - timedelta(minutes=10)
    while t <= target:
        refreshes.append(t)
        t += timedelta(minutes=refresh_every_min)
    # read slots: 30-min clock from 23:50 Nov 18, then 10-min after 04:30
    reads = []
    slot = datetime(2026, 11, 18, 23, 50, tzinfo=UTC)
    while slot < checkpoint:
        # last refresh at or before slot
        ref = max([r for r in refreshes if r <= slot])
        th = (ref - RELEASE).total_seconds() / 3600.0
        n = n_true(th)
        rates = [round(rate_true(i, th), 2) for i in range(len(trophies))]
        reads.append({"slot": slot.isoformat().replace("+00:00", "Z"),
                      "at": (slot + timedelta(milliseconds=300)).isoformat()
                      .replace("+00:00", "Z"),
                      "players_tracked": int(round(n)),
                      "trophies_earned": int(round(sum(r / 100.0 * n
                                                       for r in rates))),
                      "rates": rates})
        slot += timedelta(minutes=30 if slot < datetime(2026, 11, 19, 4, 30,
                                                        tzinfo=UTC)
                          else 10)
    case = {"id": case_id, "at": checkpoint.isoformat().replace("+00:00", "Z"),
            "horizon_minutes": horizon_min,
            "target": target.isoformat().replace("+00:00", "Z"),
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19",
                     "source": "https://www.exophase.com/game/"
                     "grand-theft-auto-vi-psn/trophies/"},
            "trophies": trophies, "history": reads, "notes": "simulated"}
    answers = [rate_true(i, (target - RELEASE).total_seconds() / 3600.0)
               for i in range(len(trophies))]
    return case, answers


def run_all():
    trophies, truth, n_true, rate_true = simulate()
    cases = []
    for cid, ck, hz, ref in [
            ("sim-p00", datetime(2026, 11, 19, 3, 0, tzinfo=UTC), 90, 240),
            ("sim-p01", datetime(2026, 11, 19, 3, 30, tzinfo=UTC), 90, 240),
            ("sim-f00", datetime(2026, 11, 21, 11, 30, tzinfo=UTC), 6510, 60)]:
        cases.append(build_case(trophies, n_true, rate_true, ck, hz, ref, cid))
    print(f"{'case':10s} {'ours':>7s} {'persist':>7s} {'curve':>7s}")
    tot = {"ours": [], "persist": [], "curve": []}
    for case, answers in cases:
        ours = prog.forecast_case(case)
        p = baseline_persistence(case)
        c = baseline_curve(case)
        s = {k: score(v, answers)
             for k, v in [("ours", ours), ("persist", p), ("curve", c)]}
        for k in tot:
            tot[k].append(s[k])
        print(f"{case['id']:10s} {s['ours']:7.3f} {s['persist']:7.3f} "
              f"{s['curve']:7.3f}")
    print("MEAN       " + " ".join(f"{sum(v)/len(v):7.3f}" for v in tot.values()))
    # coverage of ours on sim-f00
    case, answers = cases[2]
    ours = prog.forecast_case(case)
    cov = sum(1 for row, y in zip(ours["trophies"], answers)
              if row["q"][0] <= y <= row["q"][4]) / len(answers)
    med = sum(abs(y - row["q"][2]) for row, y in zip(ours["trophies"], answers)) / len(answers)
    print(f"sim-f00 ours: 90% coverage {cov:.2f}, typical miss {med:.2f} pp, "
          f"N target true {n_true(161.5):.0f}")


def practice_sensitivity():
    print("\nPractice sensitivity (assumed answers = last x 0.22, 100% stays):")
    for cid in ["p00", "p01"]:
        case = json.load(open(f"{KIT}/{cid}.in"))
        last = case["history"][-1]["rates"]
        answers = [100.0 if r == 100.0 else r * 0.22 for r in last]
        ours = prog.forecast_case(case)
        print(f"  {cid}: ours {score(ours, answers):.3f}  "
              f"persistence {score(baseline_persistence(case), answers):.3f}  "
              f"curve {score(baseline_curve(case), answers):.3f}")


if __name__ == "__main__":
    run_all()
    practice_sensitivity()
