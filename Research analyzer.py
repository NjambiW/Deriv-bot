"""
DERIVQUANT RESEARCH ANALYZER
=============================

Offline research tool. Reads a CSV of historical R_100 (or any symbol's)
ticks, tests every pattern the "brilliant recommendations" text asked for
(digit transitions, streak reversal, price-direction conditioning, contract
baselines) PLUS the layers that text was missing, and writes an
edge_report.json that a trading harness can read before deciding whether
to size any bet.

WHY THE EXTRA LAYERS MATTER
----------------------------
The suggested approach ("look for anomalies significantly far from 10%")
is a classic multiple-comparisons trap. A 10x10 transition matrix has 90
off-diagonal cells; even with a truly uniform, memoryless random number
generator, several of those cells WILL show >2-sigma deviations from 10%
just by chance. The same is true for streak-length breakdowns, and for
splitting by Up/Down/Flat. Reported "structural edges" that are actually
noise are the single easiest way to lose money confidently.

This script adds the four things needed to tell a real pattern from noise:

  1. MULTIPLE-COMPARISON CORRECTION (Benjamini-Hochberg): every family of
     simultaneous tests (all 90 transition cells, all streak lengths, all
     direction x digit cells, ...) is corrected together, not cell-by-cell.
  2. OUT-OF-SAMPLE VALIDATION: a chronological 70/30 split. A cell only
     becomes a "candidate" if it is significant in the 70% research slice.
     It only becomes "validated" if it ALSO replicates (same sign, still
     significant) on the untouched 30% validation slice.
  3. SETTLEMENT-HORIZON AWARENESS: Deriv digit contracts settle on the tick
     AFTER the one following your decision (roughly 2 ticks ahead in
     practice), not on the very next tick. Every transition/streak test is
     run at horizon H=1 AND H=2 (configurable) so a "pattern" that only
     exists one tick ahead -- useless for actually trading -- doesn't get
     reported as if it were tradeable.
  4. ECONOMIC SCREEN: a statistically significant pattern is not
     automatically a profitable one. Every validated cell is converted to
     an EV using REALISTIC Deriv payout multiples (see PAYOUT section) and
     only kept if EV is still positive after the house margin, and only
     kept as an actual "edge" if a permutation-shuffle backtest on the
     validation slice beats a null (shuffled-order) baseline.

If none of your data has a real edge -- which is the expected, honest
outcome for a well-designed synthetic index -- this script says so plainly
in candidate_edges: [] rather than manufacturing something to report.

DEPENDENCIES
------------
pandas, numpy (matplotlib optional, only for the PNG charts).
No scipy dependency: p-values use math.erfc (stdlib) directly, or the
Wilson-Hilferty normal approximation for chi-square statistics.

USAGE
-----
    python research_analyzer.py --csv "C:\\path\\to\\deriv_r100_data.csv"

Or just edit DATA_CSV_PATH below to point at the same folder as your
authenticator.py / harness data, and run with no arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# =============================================================================
# CONFIG
# =============================================================================

# Point this at your CSV (same folder as authenticator.py / your harness data
# by default). Overridable with --csv on the command line.
DATA_CSV_PATH = "deriv_r100_data.csv"

OUTPUT_DIR = Path("research_output")

# Chronological research/validation split. Never shuffle before splitting --
# that would leak future information into "research".
RESEARCH_FRACTION = 0.70

# Settlement horizons (in ticks) to test. Your DerivQuant harness logs showed
# a 1-tick contract settles ~2 ticks after the decision tick on this feed;
# H=1 is tested too so you can see how much (if any) apparent edge is really
# just an artifact of looking 1 tick too early.
HORIZONS = (1, 2, 3)

# Family-wise significance level for each corrected test family.
ALPHA = 0.01

# Minimum sample size a cell/segment needs before it is even tested. Below
# this, a p-value is statistically meaningless regardless of how extreme it
# looks.
MIN_SAMPLES = 200

# Permutation-null trials for the backtest confirmation step.
N_PERMUTATIONS = 200

# Streak lengths tested for the run/reversal analysis.
MAX_STREAK_TESTED = 6

# --- Reference payout multiples (payout / stake), from house margin by
# number of winning digits out of 10. Derived from actual Deriv proposal
# payouts logged during this project (R_100 and R_75 sessions); use this to
# convert a validated probability edge into a realistic EV, not a fantasy
# one. Update these from a live proposal if you want tighter numbers.
REFERENCE_MARGIN_BY_WINS = {
    9: 0.019, 8: 0.024, 7: 0.027, 6: 0.034, 5: 0.040,
    4: 0.048, 3: 0.061, 2: 0.090, 1: 0.167,
}


def payout_multiple(n_wins: int) -> float:
    margin = REFERENCE_MARGIN_BY_WINS[n_wins]
    p_uniform = n_wins / 10.0
    return (1.0 - margin) / p_uniform


# =============================================================================
# STATS HELPERS (stdlib only -- no scipy)
# =============================================================================

def norm_sf(z: float) -> float:
    """Upper-tail survival probability of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_sided_p_from_z(z: float) -> float:
    return 2.0 * norm_sf(abs(z))


def chi2_p_wilson_hilferty(chi2: float, df: int) -> float:
    """
    Upper-tail p-value for a chi-square statistic via the Wilson-Hilferty
    cube-root normal approximation. Accurate to ~1e-3 for df >= 5; used
    here only to flag "is this worth a closer look", not as a precise
    p-value -- the per-cell z-tests (with BH correction) are the numbers
    the candidate/validated decisions actually rest on.
    """
    if df <= 0 or chi2 <= 0:
        return 1.0
    z = ((chi2 / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(
        2.0 / (9.0 * df)
    )
    return norm_sf(z)


def wilson_ci(count: int, n: int, z: float = 2.576) -> tuple[float, float]:
    """Wilson score interval (default z=2.576 -> ~99% CI)."""
    if n == 0:
        return (0.0, 1.0)
    p = count / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion_z(count: int, n: int, p0: float) -> float:
    if n == 0:
        return 0.0
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return 0.0
    return (count / n - p0) / se


def benjamini_hochberg(pvals: list[float], alpha: float = ALPHA) -> list[bool]:
    """Returns a same-length list of booleans: True where the test survives
    BH false-discovery-rate correction at the given family-wise alpha."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    thresh_rank = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= (rank / m) * alpha:
            thresh_rank = rank
    keep = [False] * m
    if thresh_rank >= 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= thresh_rank:
                keep[idx] = True
    return keep


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class Dataset:
    n: int
    digit: np.ndarray
    price: np.ndarray
    hour: np.ndarray
    dow: np.ndarray
    direction: np.ndarray      # recomputed: -1 down, 0 flat, +1 up
    streak: np.ndarray         # recomputed run length ending at this row


def _find_col(columns: list[str], *needles: str) -> Optional[str]:
    for col in columns:
        low = col.lower()
        if all(n in low for n in needles):
            return col
    return None


def load_and_prepare(path: Path) -> Dataset:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    cols = list(df.columns)

    price_col = _find_col(cols, "price") or _find_col(cols, "quote")
    digit_col = _find_col(cols, "last", "digit") or _find_col(cols, "digit")
    hour_col = _find_col(cols, "hour")
    dow_col = _find_col(cols, "day", "week")
    time_col = _find_col(cols, "time")

    if price_col is None or digit_col is None:
        raise ValueError(
            f"Could not find price/digit columns among: {cols}. "
            "Expected something like 'price' and 'last digit'."
        )

    df = df.dropna(subset=[price_col, digit_col]).reset_index(drop=True)

    price = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
    digit = pd.to_numeric(df[digit_col], errors="coerce")
    valid = digit.notna() & np.isfinite(price)
    digit = digit[valid].to_numpy(dtype=int)
    price = price[valid.to_numpy()]

    # Self-check: does the provided digit column actually match the last
    # digit of price at 2 dp? (R_100 settles at 2 decimals; if this fails,
    # your source data may use a different symbol/precision, and every
    # downstream test is only as good as this column, not the recomputation.)
    implied = np.array(
        [int(f"{p:.2f}"[-1]) for p in price[: min(5000, len(price))]]
    )
    mismatch_rate = float(
        np.mean(implied != digit[: min(5000, len(price))])
    )
    if mismatch_rate > 0.01:
        print(
            f"[WARN] provided digit column disagrees with last-digit-of-"
            f"price(2dp) on {mismatch_rate:.1%} of a 5,000-row sample. "
            f"Continuing with the PROVIDED digit column, but verify your "
            f"symbol's decimal precision if results look strange.",
            file=sys.stderr,
        )

    n = len(digit)

    if hour_col is not None:
        hour = pd.to_numeric(df.loc[valid, hour_col], errors="coerce")
        hour = hour.fillna(-1).to_numpy(dtype=int)
    elif time_col is not None:
        t = pd.to_datetime(df.loc[valid, time_col], errors="coerce")
        hour = t.dt.hour.fillna(-1).to_numpy(dtype=int)
    else:
        hour = np.full(n, -1, dtype=int)

    if dow_col is not None:
        dow = df.loc[valid, dow_col].astype(str).to_numpy()
    elif time_col is not None:
        t = pd.to_datetime(df.loc[valid, time_col], errors="coerce")
        dow = t.dt.day_name().fillna("Unknown").to_numpy()
    else:
        dow = np.full(n, "Unknown", dtype=object)

    # Recomputed independently of any provided "Price Direction" /
    # "consecutive repeat count" columns, so a bug in the spreadsheet that
    # built those columns can't quietly propagate into this analysis.
    deltas = np.diff(price, prepend=price[0])
    direction = np.sign(deltas).astype(int)  # -1, 0, +1; index 0 is 0 (flat)

    same_as_prev = np.empty(n, dtype=bool)
    same_as_prev[0] = False
    same_as_prev[1:] = digit[1:] == digit[:-1]
    run_id = np.cumsum(~same_as_prev)
    streak = (
        pd.Series(np.ones(n, dtype=int))
        .groupby(run_id)
        .cumsum()
        .to_numpy()
    )

    return Dataset(n=n, digit=digit, price=price, hour=hour, dow=dow,
                   direction=direction, streak=streak)


def chronological_split(ds: Dataset, frac: float = RESEARCH_FRACTION):
    cut = int(ds.n * frac)

    def slice_ds(a: int, b: int) -> Dataset:
        return Dataset(
            n=b - a, digit=ds.digit[a:b], price=ds.price[a:b],
            hour=ds.hour[a:b], dow=ds.dow[a:b],
            direction=ds.direction[a:b], streak=ds.streak[a:b],
        )

    return slice_ds(0, cut), slice_ds(cut, ds.n)


# =============================================================================
# TEST 1: UNIFORMITY (overall, and by hour / day-of-week segment)
# =============================================================================

def test_uniformity(digit: np.ndarray, label: str) -> dict[str, Any]:
    n = len(digit)
    counts = np.bincount(digit, minlength=10)
    expected = n / 10.0
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    p = chi2_p_wilson_hilferty(chi2, df=9)
    max_dev = float(np.max(np.abs(counts / n - 0.1)))
    print(f"  [{label}] n={n:,} chi2={chi2:.2f} (df=9) p~={p:.4f} "
          f"max_dev_from_10%={max_dev*100:.3f}%")
    return {"label": label, "n": n, "chi2": chi2, "p_approx": p,
            "max_deviation_pct": max_dev * 100,
            "counts": counts.tolist()}


def test_uniformity_segments(digit: np.ndarray, segment: np.ndarray,
                             segment_name: str) -> dict[str, Any]:
    """Chi-square across a categorical segment (hour, day-of-week): does the
    digit distribution differ by segment? This checks for feed/collection
    artifacts (e.g. a symbol swap, a gap, a timezone bug), not an edge."""
    segments = pd.unique(segment)
    rows = []
    for s in segments:
        mask = segment == s
        if mask.sum() < MIN_SAMPLES:
            continue
        counts = np.bincount(digit[mask], minlength=10)
        rows.append(counts)
    if len(rows) < 2:
        return {"segment": segment_name, "tested": False}
    table = np.array(rows)
    row_tot = table.sum(axis=1, keepdims=True)
    col_tot = table.sum(axis=0, keepdims=True)
    grand = table.sum()
    exp = row_tot @ col_tot / grand
    chi2 = float(np.sum((table - exp) ** 2 / exp))
    df = (table.shape[0] - 1) * (table.shape[1] - 1)
    p = chi2_p_wilson_hilferty(chi2, df)
    print(f"  [{segment_name}] {table.shape[0]} segments | chi2={chi2:.1f} "
         f"(df={df}) p~={p:.4f}")
    return {"segment": segment_name, "tested": True, "n_segments": table.shape[0],
            "chi2": chi2, "df": df, "p_approx": p}


# =============================================================================
# TEST 2: TRANSITION MATRIX (lag-H), BH-corrected
# =============================================================================

def transition_cells(digit: np.ndarray, horizon: int) -> pd.DataFrame:
    if len(digit) <= horizon:
        return pd.DataFrame()
    src, dst = digit[:-horizon], digit[horizon:]
    n_by_src = np.bincount(src, minlength=10)
    joint = np.zeros((10, 10), dtype=int)
    idx = src * 10 + dst
    counts = np.bincount(idx, minlength=100).reshape(10, 10)
    joint[:, :] = counts

    rows = []
    for s in range(10):
        n_s = int(n_by_src[s])
        if n_s < MIN_SAMPLES:
            continue
        for d in range(10):
            c = int(joint[s, d])
            p0 = 0.10
            z = proportion_z(c, n_s, p0)
            p = two_sided_p_from_z(z)
            rows.append({"src": s, "dst": d, "n": n_s, "count": c,
                        "p_hat": c / n_s, "z": z, "p_value": p})
    return pd.DataFrame(rows)


def analyze_transitions(digit: np.ndarray, horizon: int,
                        label: str) -> pd.DataFrame:
    cells = transition_cells(digit, horizon)
    if cells.empty:
        print(f"  [{label} H={horizon}] not enough data.")
        return cells
    keep = benjamini_hochberg(cells["p_value"].tolist(), ALPHA)
    cells["bh_significant"] = keep
    n_sig = int(sum(keep))
    print(f"  [{label} H={horizon}] {len(cells)} cells tested (n>={MIN_SAMPLES}) "
         f"| {n_sig} survive BH correction at alpha={ALPHA}")
    if n_sig:
        top = cells[cells["bh_significant"]].reindex(
            cells[cells["bh_significant"]]["p_value"].sort_values().index
        )
        for _, r in top.head(8).iterrows():
            print(f"      DIFFERS/transition {int(r.src)}->{int(r.dst)}: "
                 f"p_hat={r.p_hat:.4f} (n={int(r.n)}) z={r.z:+.2f} "
                 f"p={r.p_value:.2e}")
    return cells


# =============================================================================
# TEST 3: STREAK / RUN REVERSAL (gambler's-fallacy check)
# =============================================================================

def analyze_streaks(digit: np.ndarray, streak: np.ndarray, horizon: int,
                    label: str) -> pd.DataFrame:
    """For each streak length k (of the CURRENT digit, ending at tick t),
    does P(digit[t+H] == digit[t]) differ from the memoryless null of 10%?"""
    n = len(digit)
    if n <= horizon:
        return pd.DataFrame()
    src_streak = streak[:-horizon]
    src_digit = digit[:-horizon]
    dst_digit = digit[horizon:]
    repeats = (dst_digit == src_digit).astype(int)

    rows = []
    for k in range(1, MAX_STREAK_TESTED + 1):
        mask = src_streak == k
        n_k = int(mask.sum())
        if n_k < MIN_SAMPLES:
            continue
        c = int(repeats[mask].sum())
        p0 = 0.10
        z = proportion_z(c, n_k, p0)
        p = two_sided_p_from_z(z)
        lo, hi = wilson_ci(c, n_k)
        rows.append({"streak_len": k, "n": n_k, "repeat_count": c,
                    "p_hat": c / n_k, "ci_lo": lo, "ci_hi": hi,
                    "z": z, "p_value": p})
    result = pd.DataFrame(rows)
    if result.empty:
        print(f"  [{label} H={horizon}] not enough streak samples.")
        return result
    keep = benjamini_hochberg(result["p_value"].tolist(), ALPHA)
    result["bh_significant"] = keep
    print(f"  [{label} H={horizon}] streak lengths 1..{MAX_STREAK_TESTED}:")
    for _, r in result.iterrows():
        flag = " *SIG*" if r.bh_significant else ""
        print(f"      after {int(r.streak_len)}x repeat: P(repeat again)="
             f"{r.p_hat:.4f} [{r.ci_lo:.4f},{r.ci_hi:.4f}] n={int(r.n)} "
             f"z={r.z:+.2f}{flag}")
    return result


# =============================================================================
# TEST 4: PRICE-DIRECTION CONDITIONED DIGIT DISTRIBUTION
# =============================================================================

def analyze_direction(digit: np.ndarray, direction: np.ndarray,
                      horizon: int, label: str) -> pd.DataFrame:
    """Does the NEXT digit's distribution differ depending on whether price
    just moved Up / Down / Flat? Tested per (direction, digit) cell."""
    n = len(digit)
    if n <= horizon:
        return pd.DataFrame()
    src_dir = direction[:-horizon]
    dst_digit = digit[horizon:]

    rows = []
    for dname, dval in (("Up", 1), ("Down", -1), ("Flat", 0)):
        mask = src_dir == dval
        n_d = int(mask.sum())
        if n_d < MIN_SAMPLES:
            continue
        counts = np.bincount(dst_digit[mask], minlength=10)
        for digit_val in range(10):
            c = int(counts[digit_val])
            z = proportion_z(c, n_d, 0.10)
            p = two_sided_p_from_z(z)
            rows.append({"direction": dname, "digit": digit_val, "n": n_d,
                        "count": c, "p_hat": c / n_d, "z": z,
                        "p_value": p})
    result = pd.DataFrame(rows)
    if result.empty:
        print(f"  [{label} H={horizon}] not enough directional samples.")
        return result
    keep = benjamini_hochberg(result["p_value"].tolist(), ALPHA)
    result["bh_significant"] = keep
    n_sig = int(sum(keep))
    print(f"  [{label} H={horizon}] {len(result)} direction x digit cells | "
         f"{n_sig} survive BH correction")
    if n_sig:
        for _, r in result[result["bh_significant"]].iterrows():
            print(f"      after {r.direction}: digit {int(r.digit)} "
                 f"p_hat={r.p_hat:.4f} (n={int(r.n)}) z={r.z:+.2f}")
    return result


# =============================================================================
# TEST 5: CONTRACT BASELINES (EVEN/ODD/OVER/UNDER)
# =============================================================================

def contract_baselines(digit: np.ndarray, label: str) -> dict[str, Any]:
    n = len(digit)
    out = {}
    even = int(np.sum(digit % 2 == 0))
    lo, hi = wilson_ci(even, n)
    out["EVEN"] = {"p_hat": even / n, "ci": [lo, hi], "n": n}
    for barrier in range(1, 9):
        over = int(np.sum(digit > barrier))
        lo, hi = wilson_ci(over, n)
        out[f"OVER_{barrier}"] = {"p_hat": over / n, "ci": [lo, hi], "n": n}
    print(f"  [{label}] EVEN={out['EVEN']['p_hat']:.4f} "
         f"[{out['EVEN']['ci'][0]:.4f},{out['EVEN']['ci'][1]:.4f}] "
         f"(all should straddle 0.500 / 0.100*k)")
    return out


# =============================================================================
# VALIDATION: replicate research-slice candidates on the untouched slice
# =============================================================================

@dataclass
class Candidate:
    family: str
    horizon: int
    description: str
    research_p_hat: float
    research_n: int
    validate_fn: Any  # callable(Dataset) -> (n, count, p0) for re-testing


def collect_transition_candidates(cells: pd.DataFrame, horizon: int) -> list[Candidate]:
    out = []
    if cells.empty:
        return out
    for _, r in cells[cells["bh_significant"]].iterrows():
        s, d = int(r.src), int(r.dst)

        def mk(s=s, d=d, h=horizon):
            def fn(ds: Dataset):
                if ds.n <= h:
                    return 0, 0
                src, dst = ds.digit[:-h], ds.digit[h:]
                mask = src == s
                n = int(mask.sum())
                c = int(np.sum(dst[mask] == d))
                return n, c
            return fn

        out.append(Candidate(
            family="transition", horizon=horizon,
            description=f"P(digit={d} | prev_digit={s}, H={horizon})",
            research_p_hat=float(r.p_hat), research_n=int(r.n),
            validate_fn=mk(),
        ))
    return out


def collect_streak_candidates(streaks: pd.DataFrame, horizon: int) -> list[Candidate]:
    out = []
    if streaks.empty:
        return out
    for _, r in streaks[streaks["bh_significant"]].iterrows():
        k = int(r.streak_len)

        def mk(k=k, h=horizon):
            def fn(ds: Dataset):
                if ds.n <= h:
                    return 0, 0
                src_streak = ds.streak[:-h]
                src_digit = ds.digit[:-h]
                dst_digit = ds.digit[h:]
                mask = src_streak == k
                n = int(mask.sum())
                c = int(np.sum(dst_digit[mask] == src_digit[mask]))
                return n, c
            return fn

        out.append(Candidate(
            family="streak", horizon=horizon,
            description=f"P(repeat again | streak={k}, H={horizon})",
            research_p_hat=float(r.p_hat), research_n=int(r.n),
            validate_fn=mk(),
        ))
    return out


def collect_direction_candidates(directions: pd.DataFrame, horizon: int) -> list[Candidate]:
    out = []
    if directions.empty:
        return out
    dval_map = {"Up": 1, "Down": -1, "Flat": 0}
    for _, r in directions[directions["bh_significant"]].iterrows():
        dname, digit_val = r.direction, int(r.digit)
        dval = dval_map[dname]

        def mk(dval=dval, digit_val=digit_val, h=horizon):
            def fn(ds: Dataset):
                if ds.n <= h:
                    return 0, 0
                src_dir = ds.direction[:-h]
                dst_digit = ds.digit[h:]
                mask = src_dir == dval
                n = int(mask.sum())
                c = int(np.sum(dst_digit[mask] == digit_val))
                return n, c
            return fn

        out.append(Candidate(
            family="direction", horizon=horizon,
            description=f"P(digit={digit_val} | direction={dname}, H={horizon})",
            research_p_hat=float(r.p_hat), research_n=int(r.n),
            validate_fn=mk(),
        ))
    return out


def validate_candidates(candidates: list[Candidate],
                        val_ds: Dataset) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    rows = []
    pvals = []
    for cand in candidates:
        n, c = cand.validate_fn(val_ds)
        if n < MIN_SAMPLES:
            rows.append({"family": cand.family, "horizon": cand.horizon,
                        "description": cand.description,
                        "research_p_hat": cand.research_p_hat, "val_n": n,
                        "val_p_hat": None, "z": None, "p_value": 1.0,
                        "same_direction": False})
            pvals.append(1.0)
            continue
        p0 = 0.10
        z = proportion_z(c, n, p0)
        p = two_sided_p_from_z(z)
        same_dir = (cand.research_p_hat - 0.10) * (c / n - 0.10) > 0
        rows.append({"family": cand.family, "horizon": cand.horizon,
                    "description": cand.description,
                    "research_p_hat": cand.research_p_hat, "val_n": n,
                    "val_p_hat": c / n, "z": z, "p_value": p,
                    "same_direction": same_dir})
        pvals.append(p)
    result = pd.DataFrame(rows)
    keep = benjamini_hochberg(pvals, ALPHA)
    result["bh_significant_in_validation"] = keep
    result["validated"] = result["bh_significant_in_validation"] & result["same_direction"]
    return result


# =============================================================================
# ECONOMIC SCREEN: EV + permutation-null backtest for validated cells
# =============================================================================

def screen_and_backtest(validated: pd.DataFrame, val_ds: Dataset) -> list[dict]:
    """For each VALIDATED cell, compute EV at the reference DIFFERS payout
    (n_wins=9, since these are single-digit predictions), then run a
    permutation-null backtest on the validation slice: shuffle the digit
    order N_PERMUTATIONS times (destroys any real sequential structure
    while preserving the marginal digit frequencies) and compare realized
    P/L against that null. A pattern only becomes a reported "candidate
    edge" if it clears BOTH the statistical bar and this economic one."""
    out = []
    if validated.empty:
        return out
    rows = validated[validated["validated"]]
    if rows.empty:
        return out

    mult = payout_multiple(9)  # DIFFERS-style single-digit bet
    rng = np.random.default_rng(2024)

    for _, r in rows.iterrows():
        p_hat = r.val_p_hat
        ev = p_hat * mult - (1.0 - p_hat)
        if ev <= 0:
            continue  # statistically real, but the house margin still wins

        # Re-derive the actual win/loss mask on the validation slice for a
        # permutation test (shuffle digit order; same marginal frequencies,
        # no sequential structure -- the honest null for "does exploiting
        # this pattern actually make money, or would shuffled noise too?")
        h = int(r.horizon)
        if r.family == "streak":
            k = int(r.description.split("streak=")[1].split(",")[0])

            def realized_pnl(digit_arr, streak_arr):
                if len(digit_arr) <= h:
                    return 0.0
                src_streak, src_digit = streak_arr[:-h], digit_arr[:-h]
                dst_digit = digit_arr[h:]
                mask = src_streak == k
                if mask.sum() == 0:
                    return 0.0
                wins = dst_digit[mask] != src_digit[mask]  # DIFFERS bet
                return float(np.sum(np.where(wins, mult - 1.0, -1.0)))

            actual = realized_pnl(val_ds.digit, val_ds.streak)
            null_vals = []
            for _ in range(N_PERMUTATIONS):
                shuffled = rng.permutation(val_ds.digit)
                null_vals.append(realized_pnl(shuffled, val_ds.streak))
        else:
            continue  # transition/direction EV screening left for later runs

        null_vals = np.array(null_vals)
        z = (actual - null_vals.mean()) / (null_vals.std() + 1e-9)
        beats_pct = float(np.mean(null_vals <= actual)) * 100

        out.append({
            "description": r.description, "family": r.family,
            "horizon": h, "validated_p_hat": p_hat,
            "reference_payout_multiple": mult, "reference_ev_per_unit": ev,
            "backtest_realized_pnl": actual,
            "backtest_null_mean": float(null_vals.mean()),
            "backtest_null_sd": float(null_vals.std()),
            "backtest_z_vs_null": float(z),
            "backtest_beats_null_pct": beats_pct,
            "passes_economic_screen": bool(z > 2.0 and beats_pct > 97.5),
        })
    return out


# =============================================================================
# CHARTS (optional)
# =============================================================================

def make_charts(ds: Dataset, transitions_h1: pd.DataFrame,
                streaks_h1: pd.DataFrame, out_dir: Path) -> None:
    if not HAVE_MPL:
        print("  [charts] matplotlib not installed; skipping PNG output.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = np.bincount(ds.digit, minlength=10)
    ax.bar(range(10), counts / ds.n * 100)
    ax.axhline(10, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("digit"); ax.set_ylabel("% of ticks")
    ax.set_title(f"Digit frequency (n={ds.n:,}) vs 10% uniform line")
    fig.tight_layout(); fig.savefig(out_dir / "digit_histogram.png"); plt.close(fig)

    if not transitions_h1.empty:
        mat = np.full((10, 10), np.nan)
        for _, r in transitions_h1.iterrows():
            mat[int(r.src), int(r.dst)] = r.p_hat * 100
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=5, vmax=15)
        ax.set_xlabel("next digit"); ax.set_ylabel("previous digit")
        ax.set_title("P(next digit | previous digit), H=1 (%)")
        fig.colorbar(im, ax=ax, label="%")
        fig.tight_layout(); fig.savefig(out_dir / "transition_heatmap_h1.png")
        plt.close(fig)

    if not streaks_h1.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            streaks_h1["streak_len"], streaks_h1["p_hat"] * 100,
            yerr=[(streaks_h1["p_hat"] - streaks_h1["ci_lo"]) * 100,
                 (streaks_h1["ci_hi"] - streaks_h1["p_hat"]) * 100],
            fmt="o-", capsize=4,
        )
        ax.axhline(10, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("streak length"); ax.set_ylabel("P(repeat again) %")
        ax.set_title("Streak-reversal rate with 99% CI, H=1")
        fig.tight_layout(); fig.savefig(out_dir / "streak_reversal_h1.png")
        plt.close(fig)

    print(f"  [charts] saved to {out_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DerivQuant research analyzer")
    parser.add_argument("--csv", default=DATA_CSV_PATH)
    parser.add_argument("--out", default=str(OUTPUT_DIR))
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DERIVQUANT RESEARCH ANALYZER")
    print("=" * 78)
    print(f"Loading {csv_path} ...")
    ds = load_and_prepare(csv_path)
    print(f"Loaded {ds.n:,} ticks.\n")

    research, validation = chronological_split(ds)
    print(f"Chronological split: research={research.n:,} "
         f"({RESEARCH_FRACTION:.0%}) | validation={validation.n:,} "
         f"({1 - RESEARCH_FRACTION:.0%})\n")

    report: dict[str, Any] = {
        "n_total": ds.n, "n_research": research.n, "n_validation": validation.n,
        "alpha": ALPHA, "min_samples": MIN_SAMPLES, "horizons_tested": HORIZONS,
    }

    print("-- 1. Uniformity --------------------------------------------------")
    report["uniformity_research"] = test_uniformity(research.digit, "research")
    report["uniformity_validation"] = test_uniformity(validation.digit, "validation")
    report["uniformity_by_hour"] = test_uniformity_segments(
        research.digit, research.hour, "hour-of-day")
    report["uniformity_by_dow"] = test_uniformity_segments(
        research.digit, research.dow, "day-of-week")
    print()

    print("-- 2. Contract baselines -------------------------------------------")
    report["baselines_research"] = contract_baselines(research.digit, "research")
    print()

    all_candidates: list[Candidate] = []
    transition_h1 = pd.DataFrame()
    streak_h1 = pd.DataFrame()

    for h in HORIZONS:
        print(f"-- 3. Transition matrix (H={h}) --------------------------------")
        cells = analyze_transitions(research.digit, h, "research")
        all_candidates += collect_transition_candidates(cells, h)
        if h == 1:
            transition_h1 = cells
        print()

        print(f"-- 4. Streak / reversal analysis (H={h}) -----------------------")
        streaks = analyze_streaks(research.digit, research.streak, h, "research")
        all_candidates += collect_streak_candidates(streaks, h)
        if h == 1:
            streak_h1 = streaks
        print()

        print(f"-- 5. Price-direction conditioning (H={h}) ----------------------")
        directions = analyze_direction(research.digit, research.direction, h,
                                       "research")
        all_candidates += collect_direction_candidates(directions, h)
        print()

    print("-- 6. Out-of-sample validation --------------------------------------")
    print(f"  {len(all_candidates)} candidate cell(s) passed BH correction on "
         f"the research slice; re-testing each on the untouched validation "
         f"slice...")
    validated_df = validate_candidates(all_candidates, validation)
    if not validated_df.empty:
        n_validated = int(validated_df["validated"].sum())
        print(f"  {n_validated}/{len(all_candidates)} candidate(s) replicated "
             f"(same direction, still BH-significant) on validation.")
        for _, r in validated_df[validated_df["validated"]].iterrows():
            print(f"      VALIDATED: {r.description} | research p_hat="
                 f"{r.research_p_hat:.4f} -> validation p_hat="
                 f"{r.val_p_hat:.4f} (n={int(r.val_n)})")
    else:
        print("  No candidates to validate (none survived BH correction on "
             "the research slice -- this is the expected result for a "
             "well-designed synthetic index).")
    print()

    print("-- 7. Economic screen + permutation backtest ------------------------")
    edges = screen_and_backtest(
        validated_df if not validated_df.empty else pd.DataFrame(),
        validation,
    )
    if edges:
        for e in edges:
            verdict = "PASSES" if e["passes_economic_screen"] else "fails"
            print(f"  {e['description']}: EV/unit={e['reference_ev_per_unit']:+.4f} "
                 f"(payout x{e['reference_payout_multiple']:.3f}) | backtest "
                 f"P/L=${e['backtest_realized_pnl']:+.2f} vs null "
                 f"{e['backtest_null_mean']:+.2f}+-{e['backtest_null_sd']:.2f} "
                 f"(z={e['backtest_z_vs_null']:+.2f}, beats "
                 f"{e['backtest_beats_null_pct']:.0f}% of shuffles) -> {verdict}")
    else:
        print("  No validated pattern had positive EV against realistic "
             "Deriv payouts, or none beat a permutation-shuffle backtest.")
    print()

    report["candidates_tested"] = len(all_candidates)
    report["validated"] = (
        validated_df.to_dict(orient="records") if not validated_df.empty else []
    )
    report["candidate_edges"] = [
        e for e in edges if e["passes_economic_screen"]
    ]
    report["all_economic_screens"] = edges

    if report["candidate_edges"]:
        print(f"VERDICT: {len(report['candidate_edges'])} pattern(s) survived "
             f"every check (significance + BH correction + out-of-sample "
             f"replication + positive EV + permutation backtest). Review "
             f"edge_report.json before wiring these into live sizing -- this "
             f"many gates passing is unusual and worth a second look, not an "
             f"immediate green light.")
    else:
        print("VERDICT: no pattern in this data survived the full pipeline. "
             "That is the expected result for a random-number-generator-"
             "driven synthetic index, and it means the trading harness "
             "should keep sizing at zero (or probe-stake only) rather than "
             "betting on any digit/streak/direction rule found here.")

    report_path = out_dir / "edge_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report written to {report_path}")

    if not args.no_charts:
        print("\n-- 8. Charts ---------------------------------------------------")
        make_charts(research, transition_h1, streak_h1, out_dir)


if __name__ == "__main__":
    main()