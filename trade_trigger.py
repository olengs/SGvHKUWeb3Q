"""
trade_trigger.py
================
Pure signal module — no order execution.

Public API
----------
    signal, detail = evaluate(today_bands, position=None)

    Returns
    -------
    signal : int
         1  → BUY   (flat, all entry conditions met)
         0  → HOLD  (in position, keep; or flat, conditions not met)
        -1  → SELL  (in position: profit trigger + low EV, or negative EV cut)

    detail : dict
        Full breakdown: skew, mc metrics, action label, reason string.

Decision logic
--------------
ENTRY (position=None):
    1. CDF skew: P(end > S0) / P(end < S0) >= SKEW_THRESHOLD
       Derived from calibrated Normal CDF — not from raw bracket midpoints.
    2. EV (expected end-of-day log-return) >= EV_MIN_ENTRY  (2.5%)
    3. P(price drops > $2,000 within 2 hours) < MAX_DRAWDOWN_PROB (10%)
    All three pass → signal = 1

HOLD / SELL / CUT (position dict supplied):
    EV < 0                                → signal = -1  (CUT, full derisking)
    position up >= 2% AND EV < EV_MIN_HOLD→ signal = -1  (SELL, take profit)
    otherwise                             → signal =  0  (HOLD)
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SKEW_THRESHOLD     = 1.03   # P_bull / P_bear ratio required for entry
EV_MIN_ENTRY       = 0.003  # 0.3%  — min end-of-day EV to open
EV_MIN_HOLD        = 0.010  # 1.0%  — min EV to keep holding after profit trigger
MAX_DRAWDOWN_PROB  = 0.10   # 10%   — max P(2h drop > $2,000)
PROFIT_TRIGGER_PCT = 0.02   # 2%    — unrealised gain % that activates profit check
MC_PATHS           = 20_000
MC_SEED            = 42
TRADING_HOURS_DAY  = 24


# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ──────────────────────────────────────────────────────────────────────────────

def _bracket_probs_normal(S0, brackets, mu_r, sigma_r):
    """Analytic P(price in bracket) under Normal log-return."""
    probs = np.zeros(len(brackets))
    for i, (_, lo, hi, _) in enumerate(brackets):
        lo_r  = np.log(lo / S0) if lo is not None else -np.inf
        hi_r  = np.log(hi / S0) if hi is not None else  np.inf
        cdf_h = 1.0 if hi_r ==  np.inf else norm.cdf(hi_r, mu_r, sigma_r)
        cdf_l = 0.0 if lo_r == -np.inf else norm.cdf(lo_r, mu_r, sigma_r)
        probs[i] = max(cdf_h - cdf_l, 0.0)
    return probs / max(probs.sum(), 1e-9)


def _kl(p, q, eps=1e-9):
    p = np.clip(p, eps, 1); p /= p.sum()
    q = np.clip(q, eps, 1); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _calibrate(S0, brackets):
    """
    Fit Normal(mu_r, sigma_r) to Polymarket brackets via KL-minimisation.
    Grid warm-start → L-BFGS-B fine-tune.
    Returns (mu_r, sigma_r) — daily log-return params.
    """
    raw    = np.array([b[3] for b in brackets], dtype=float)
    target = raw / raw.sum()

    def obj(params):
        return _kl(target, _bracket_probs_normal(S0, brackets, params[0], np.exp(params[1])))

    best_kl, best_x = np.inf, None
    for mu in np.linspace(-0.08, 0.04, 25):
        for ls in np.linspace(np.log(0.005), np.log(0.10), 25):
            k = _kl(target, _bracket_probs_normal(S0, brackets, mu, np.exp(ls)))
            if k < best_kl:
                best_kl, best_x = k, (mu, ls)

    res = minimize(obj, best_x, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-11})
    return float(res.x[0]), float(np.exp(res.x[1]))


# ──────────────────────────────────────────────────────────────────────────────
# SKEW  (CDF-based, split at S0)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_skew(mu_r, sigma_r):
    """
    P(end > S0) = 1 - Φ(0 ; mu_r, sigma_r)
    P(end < S0) =     Φ(0 ; mu_r, sigma_r)

    Using the continuous CDF avoids any bracket-boundary alignment issues —
    S0 is always the exact split regardless of where Polymarket drew its lines.
    """
    p_bear = float(norm.cdf(0.0, mu_r, sigma_r))
    p_bull = 1.0 - p_bear
    skew   = p_bull / max(p_bear, 1e-9)
    return {
        "p_bull":     p_bull,
        "p_bear":     p_bear,
        "skew_ratio": skew,
        "is_bullish": skew >= SKEW_THRESHOLD,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MONTE CARLO METRICS
# ──────────────────────────────────────────────────────────────────────────────

def _run_mc(S0, mu_r, sigma_r):
    """Single end-of-day GBM step. Returns final prices (MC_PATHS,)."""
    rng = np.random.default_rng(MC_SEED)
    Z   = rng.standard_normal(MC_PATHS)
    return S0 * np.exp((mu_r - 0.5 * sigma_r**2) + sigma_r * Z)


def _compute_mc_metrics(S0, mu_r, sigma_r):
    """
    Returns end-of-day EV, CVaR-5%, P(up), median,
    plus P(2h intraday drop > $2,000).

    2-hour scaling: sigma_2h = sigma_r * sqrt(2/24)
                    mu_2h    = mu_r    *       2/24
    """
    # End-of-day
    eod      = _run_mc(S0, mu_r, sigma_r)
    ev       = float(np.mean(np.log(eod / S0)))
    prob_up  = float(np.mean(eod > S0))
    median   = float(np.median(eod))
    tail     = eod[eod <= np.percentile(eod, 5)]
    cvar_5   = float((S0 - tail.mean()) / S0) if len(tail) else 0.0

    # 2-hour intraday
    dt       = 2.0 / TRADING_HOURS_DAY
    intra    = _run_mc(S0, mu_r * dt, sigma_r * np.sqrt(dt))
    p_drop2k = float(np.mean(intra < S0 - 2_000))

    return {
        "ev":           ev,
        "cvar_5pct":    cvar_5,
        "prob_up":      prob_up,
        "median":       median,
        "p_2h_drop_2k": p_drop2k,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(today_bands: list, position: dict | None = None, S0: float | None = None) -> tuple[int, dict]:
    """
    Run the full analysis pipeline and return a single integer signal.

    Parameters
    ----------
    today_bands : list of (label, lo, hi, prob)
        Polymarket bracket data for today.
    position : dict | None
        None  → flat, evaluating entry.
        dict  → {"entry_price": float, "qty": float}  open long position.
    S0 : float | None
        Current BTC spot price. If None, must be injected by the caller
        (keeps this module free of any exchange API dependency).

    Returns
    -------
    signal : int
         1 = BUY
         0 = HOLD / SKIP
        -1 = SELL / CUT

    detail : dict
        {
            "signal":   int,
            "action":   str,   # "BUY"|"HOLD"|"SKIP"|"SELL"|"CUT"
            "reason":   str,
            "skew":     dict,
            "mc":       dict,
            "mu_r":     float,
            "sigma_r":  float,
        }
    """
    if S0 is None:
        raise ValueError("S0 (current BTC price) must be provided to evaluate()")

    # ── Step 1: calibrate Normal to Polymarket bands ──────────────────────────
    mu_r, sigma_r = _calibrate(S0, today_bands)

    # ── Step 2: CDF skew at S0 ───────────────────────────────────────────────
    skew = _compute_skew(mu_r, sigma_r)

    # ── Step 3: MC metrics ────────────────────────────────────────────────────
    mc = _compute_mc_metrics(S0, mu_r, sigma_r)

    # ── Step 4: decision ──────────────────────────────────────────────────────
    if position is None:
        # FLAT — check entry conditions
        fails = []
        if not skew["is_bullish"]:
            fails.append(
                f"skew {skew['skew_ratio']:.2f}x < {SKEW_THRESHOLD}x "
                f"[bull={skew['p_bull']:.1%} bear={skew['p_bear']:.1%}]"
            )
        if mc["ev"] < EV_MIN_ENTRY:
            fails.append(f"EV {mc['ev']*100:+.2f}% < {EV_MIN_ENTRY*100:.1f}%")
        if mc["p_2h_drop_2k"] > MAX_DRAWDOWN_PROB:
            fails.append(
                f"P(2h drop>$2k) {mc['p_2h_drop_2k']:.1%} > {MAX_DRAWDOWN_PROB:.0%}"
            )

        if not fails:
            signal, action = 1, "BUY"
            reason = (
                f"All entry conditions met | "
                f"skew={skew['skew_ratio']:.2f}x  "
                f"EV={mc['ev']*100:+.2f}%  "
                f"P(2h drop>$2k)={mc['p_2h_drop_2k']:.1%}  "
                f"P(up)={mc['prob_up']:.1%}"
            )
        else:
            signal, action = 0, "SKIP"
            reason = "Entry blocked: " + " | ".join(fails)

    else:
        # IN POSITION — check cut / profit
        entry_price    = float(position["entry_price"])
        unrealised_pct = (S0 - entry_price) / entry_price

        if mc["ev"] < -0.01:
            signal, action = -1, "CUT"
            reason = (
                f"EV {mc['ev']*100:+.2f}% below -1% — full derisking "
                f"[PnL {unrealised_pct*100:+.2f}%]"
            )
        elif unrealised_pct >= PROFIT_TRIGGER_PCT and mc["ev"] < EV_MIN_HOLD:
            signal, action = -1, "SELL"
            reason = (
                f"Profit trigger +{unrealised_pct*100:.2f}% hit; "
                f"EV {mc['ev']*100:+.2f}% < hold min {EV_MIN_HOLD*100:.1f}%"
            )
        else:
            signal, action = 0, "HOLD"
            if unrealised_pct >= PROFIT_TRIGGER_PCT:
                reason = (
                    f"Profit trigger hit but EV {mc['ev']*100:+.2f}% "
                    f">= {EV_MIN_HOLD*100:.1f}% — holding"
                )
            else:
                reason = (
                    f"PnL {unrealised_pct*100:+.2f}% below "
                    f"{PROFIT_TRIGGER_PCT*100:.0f}% trigger — holding"
                )

    detail = {
        "signal":   signal,
        "action":   action,
        "reason":   reason,
        "skew":     skew,
        "mc":       mc,
        "mu_r":     mu_r,
        "sigma_r":  sigma_r,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    signal_label = {1: "BUY  (+1)", 0: "HOLD ( 0)", -1: "SELL (-1)"}[signal]
    print("\n" + "=" * 60)
    print(f"  SIGNAL: {signal_label}  |  Action: {action}")
    print(f"  Reason: {reason}")
    print(f"  {'─' * 56}")
    print(f"  Calibrated   mu_r={mu_r*100:+.4f}%  "
          f"sigma_r={sigma_r*100:.4f}%  "
          f"Ann.Vol={sigma_r*(365**0.5)*100:.1f}%")
    print(f"  Skew         P_bull={skew['p_bull']:.1%}  "
          f"P_bear={skew['p_bear']:.1%}  "
          f"ratio={skew['skew_ratio']:.2f}x  "
          f"({'✅' if skew['is_bullish'] else '❌'})")
    print(f"  MC           EV={mc['ev']*100:+.2f}%  "
          f"P(up)={mc['prob_up']:.1%}  "
          f"Median=${mc['median']:,.0f}  "
          f"CVaR-5%={mc['cvar_5pct']:.2%}")
    print(f"  Drawdown     P(2h drop>$2k)={mc['p_2h_drop_2k']:.1%}")
    print("=" * 60 + "\n")

    return signal, detail


if __name__ == "__main__":
    import json
    import re
    from polymarket import PolyMarket, fetch_polymarket_bands
    from datetime import datetime
    import yfinance as yf
    raw = yf.download("BTC-USD", period="1d", interval="1m", progress=False)
    S0  = float(raw["Close"].iloc[-1])
    print(f"[TEST] Live S0 = ${S0:,.2f}")

    # ── Fetch live Polymarket bands ───────────────────────────────────────────
    market = PolyMarket()
    _, bands = fetch_polymarket_bands(market)

    with open("polymarket_bands.json", "w") as f:
        json.dump(bands, f, indent=2)

    today_key = f"{datetime.now().strftime('%b')} {int(datetime.now().strftime('%d'))}"
    if today_key not in bands or not bands[today_key]:
        print(f"[TEST] No bands found for {today_key} — exiting")
        exit()

    def _parse_label(label):
        nums = [int(n) * 1000 for n in re.findall(r"\d+", label)]
        if label.startswith("<"): return None, nums[0]
        if label.startswith(">"): return nums[0], None
        if len(nums) >= 2:        return nums[0], nums[1]
        return None, None

    today_bands = [
        (b["b"], *_parse_label(b["b"]), float(b["p"]))
        for b in bands[today_key]
    ]
    print(f"[TEST] {len(today_bands)} brackets loaded for {today_key}\n")

    # ── Evaluate — flat (no open position) ───────────────────────────────────
    signal, detail = evaluate(today_bands, position=None, S0=S0)
    print(f"  → signal = {signal}")