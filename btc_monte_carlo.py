"""
BTC Monte Carlo — Normal Distribution + Polymarket Calibration
==============================================================
Architecture:
    1.  Fetch 3Y BTC-USD history via yfinance
    2.  Engineer technical features, train LightGBM (drift baseline)
    3.  For each Polymarket day:
          a. Fit a Normal(mu_r, sigma_r) on log-returns by minimising
             KL-divergence between analytic bracket CDFs and Polymarket odds
          b. Grid search warm-start → L-BFGS-B fine-tune (no MC noise)
          c. Run 20k-path GBM with calibrated (mu_r, sigma_r)
    4.  Output: per-day bracket comparison table + combined chart

Why Normal instead of NIG:
    NIG collapses to delta → 0 when optimised against sharp bracket
    distributions (a known numerical pathology). Plain Normal is
    analytically tractable, stable, and achieves GOOD/EXCELLENT KL
    on real Polymarket data for BTC 1-7 day horizons.

Requirements:
    pip install yfinance lightgbm scikit-learn pandas numpy matplotlib scipy
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
import json
from binformat import bands_by_date_format
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TICKER   = "BTC-USD"
LOOKBACK = 3
PATHS    = 20_000
SEED     = 42

LGB_PARAMS = {
    "objective": "regression", "metric": "mae",
    "n_estimators": 500, "learning_rate": 0.03,
    "num_leaves": 31, "max_depth": 5,
    "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "min_child_samples": 20,
    "lambda_l1": 0.1, "lambda_l2": 0.1,
    "verbose": -1, "random_state": SEED,
}

# Live Polymarket CLOB odds (label, lo, hi, probability)
# lo=None → open lower tail   hi=None → open upper tail

with open("polymarket_bands.json", "r") as f:
    my_json = json.load(f)

POLYMARKET = bands_by_date_format(my_json)

# ─────────────────────────────────────────────────────────────
# NORMAL-DISTRIBUTION BRACKET CALIBRATOR
# ─────────────────────────────────────────────────────────────

def bracket_probs_normal(S0, brackets, mu_r, sigma_r):
    """
    P(price in bracket) = Phi(log(hi/S0); mu_r, sigma_r)
                        - Phi(log(lo/S0); mu_r, sigma_r)
    Exact, analytic — no Monte Carlo required.
    """
    probs = np.zeros(len(brackets))
    for i, (_, lo, hi, _) in enumerate(brackets):
        lo_r = np.log(lo / S0) if lo is not None else -np.inf
        hi_r = np.log(hi / S0) if hi is not None else  np.inf
        cdf_hi = 1.0 if hi_r ==  np.inf else norm.cdf(hi_r, mu_r, sigma_r)
        cdf_lo = 0.0 if lo_r == -np.inf else norm.cdf(lo_r, mu_r, sigma_r)
        probs[i] = max(cdf_hi - cdf_lo, 0.0)
    total = probs.sum()
    return probs / max(total, 1e-9)


def kl_divergence(p, q, eps=1e-9):
    """KL(p_target || q_model)"""
    p = np.clip(p, eps, 1); p /= p.sum()
    q = np.clip(q, eps, 1); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def calibrate_normal(S0, brackets, verbose=True):
    """
    Find (mu_r, sigma_r) for Normal log-return distribution that
    minimises KL-divergence against Polymarket bracket probabilities.

    Returns: mu_r, sigma_r, kl_final, model_probs, target_probs
    """
    raw    = np.array([b[3] for b in brackets])
    target = raw / raw.sum()

    def objective(params):
        mu_r, log_sigma = params
        sigma_r = np.exp(log_sigma)
        q = bracket_probs_normal(S0, brackets, mu_r, sigma_r)
        return kl_divergence(target, q)

    # ── Grid search warm-start ──
    # Wide search range: mu from -8% to +4%, sigma from 0.5% to 10%
    best_kl, best_x = np.inf, None
    for mu_r in np.linspace(-0.08, 0.04, 35):
        for log_s in np.linspace(np.log(0.005), np.log(0.10), 35):
            sigma_r = np.exp(log_s)
            q = bracket_probs_normal(S0, brackets, mu_r, sigma_r)
            k = kl_divergence(target, q)
            if k < best_kl:
                best_kl = k
                best_x  = (mu_r, log_s)

    if verbose:
        print(f"[CALIB] Grid warm-start: KL={best_kl:.6f}  "
              f"mu={best_x[0]*100:+.4f}%  sigma={np.exp(best_x[1])*100:.4f}%")

    # ── L-BFGS-B fine-tune ──
    result = minimize(
        objective, best_x, method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-12},
    )
    mu_r    = result.x[0]
    sigma_r = np.exp(result.x[1])
    kl_fin  = result.fun
    q_final = bracket_probs_normal(S0, brackets, mu_r, sigma_r)

    if verbose:
        grade = ("EXCELLENT" if kl_fin < 0.005 else
                 "GOOD"      if kl_fin < 0.020 else
                 "FAIR"      if kl_fin < 0.100 else "POOR")
        print(f"[CALIB] Optimised:  KL={kl_fin:.8f}  [{grade}]")
        print(f"[CALIB] mu_r={mu_r*100:+.4f}%  sigma_r={sigma_r*100:.4f}%")
        print(f"[CALIB] Interpretation:")
        print(f"         Daily drift (mu_r):    {mu_r*100:+.4f}%")
        print(f"         Daily vol  (sigma_r):  {sigma_r*100:.4f}%")
        print(f"         Annualised vol:         {sigma_r*np.sqrt(365)*100:.1f}%")
        print()

    return mu_r, sigma_r, kl_fin, q_final, target


def print_bracket_report(brackets, target, model_probs, label, kl):
    W = 80
    grade = ("EXCELLENT" if kl < 0.005 else
             "GOOD"      if kl < 0.020 else
             "FAIR"      if kl < 0.100 else "POOR")
    print("=" * W)
    print(f"  {label}  —  KL={kl:.8f}  [{grade}]")
    print(f"  {'Bracket':>8}  {'Polymarket':>11}  {'Model':>10}  {'Diff':>8}  {'Pass':>5}")
    print(f"  {'-'*(W-2)}")
    for i, (lbl, _, _, _) in enumerate(brackets):
        pm, md = target[i], model_probs[i]
        diff   = md - pm
        ok     = "✓" if abs(diff) <= 0.030 else ("~" if abs(diff) <= 0.060 else "✗")
        print(f"  {lbl:>8}  {pm:>10.2%}  {md:>10.2%}  {diff:>+7.2%}  {ok:>5}")
    rmse = np.sqrt(np.mean((model_probs - target)**2))
    print(f"  {'-'*(W-2)}")
    print(f"  RMSE={rmse:.5f}   KL={kl:.8f}   [{grade}]")
    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────
# MONTE CARLO with calibrated normal shocks
# ─────────────────────────────────────────────────────────────

def run_mc(S0, mu_r, sigma_r, sim_days, paths, seed=SEED):
    """
    GBM with Normal(mu_r, sigma_r) log-return shocks.
    Returns sims of shape (sim_days+1, paths).
    """
    rng  = np.random.default_rng(seed)
    sims = np.empty((sim_days + 1, paths))
    sims[0] = S0
    for t in range(sim_days):
        Z       = rng.standard_normal(paths)
        drift   = mu_r - 0.5 * sigma_r**2
        sims[t+1] = sims[t] * np.exp(drift + sigma_r * Z)
    return sims


# ─────────────────────────────────────────────────────────────
# DATA + ML (feature engineering + LightGBM drift baseline)
# ─────────────────────────────────────────────────────────────

def fetch_data(ticker, lookback_years):
    end   = datetime.today()
    start = end - timedelta(days=lookback_years * 365)
    print(f"[DATA] Fetching {ticker} ({start.date()} → {end.date()}) ...")
    raw = yf.download(ticker, start=start, end=end, progress=False)
    df  = raw[["Open","High","Low","Close","Volume"]].copy()
    df.columns = ["open","high","low","close","volume"]
    df.dropna(inplace=True)
    print(f"[DATA] {len(df)} days loaded.\n")
    return df


def make_features(df):
    d = df.copy()
    d["log_ret"]    = np.log(d["close"] / d["close"].shift(1))
    d["log_ret_2"]  = np.log(d["close"] / d["close"].shift(2))
    d["log_ret_5"]  = np.log(d["close"] / d["close"].shift(5))
    d["log_ret_10"] = np.log(d["close"] / d["close"].shift(10))
    d["log_ret_20"] = np.log(d["close"] / d["close"].shift(20))
    d["vol_5"]      = d["log_ret"].rolling(5).std()
    d["vol_10"]     = d["log_ret"].rolling(10).std()
    d["vol_20"]     = d["log_ret"].rolling(20).std()
    d["vol_ratio"]  = d["vol_5"] / (d["vol_20"] + 1e-9)
    d["sma_5"]      = d["close"].rolling(5).mean()
    d["sma_20"]     = d["close"].rolling(20).mean()
    d["sma_50"]     = d["close"].rolling(50).mean()
    d["price_vs_sma5"]  = d["close"] / (d["sma_5"]  + 1e-9) - 1
    d["price_vs_sma20"] = d["close"] / (d["sma_20"] + 1e-9) - 1
    d["price_vs_sma50"] = d["close"] / (d["sma_50"] + 1e-9) - 1
    d["sma5_vs_sma20"]  = d["sma_5"] / (d["sma_20"] + 1e-9) - 1
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi_norm"]  = (100 - 100/(1 + gain/(loss+1e-9)) - 50) / 50
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    d["macd_norm"] = (macd - macd.ewm(span=9,adjust=False).mean()) / (d["close"]+1e-9)
    bb_mid = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_pos"]     = (d["close"] - (bb_mid-2*bb_std)) / (4*bb_std+1e-9)
    d["hl_range"]   = (d["high"] - d["low"]) / (d["close"]+1e-9)
    d["hl_range_5"] = d["hl_range"].rolling(5).mean()
    d["vol_ratio_v"]= d["volume"] / (d["volume"].rolling(20).mean()+1e-9)
    d["mom_5"]      = d["close"] / (d["close"].shift(5) +1e-9) - 1
    d["mom_10"]     = d["close"] / (d["close"].shift(10)+1e-9) - 1
    d["mom_20"]     = d["close"] / (d["close"].shift(20)+1e-9) - 1
    d["dow"]        = d.index.dayofweek.astype(float) / 6.0
    d["target"]     = d["log_ret"].shift(-1)
    return d.dropna()


FEAT_COLS = [
    "log_ret","log_ret_2","log_ret_5","log_ret_10","log_ret_20",
    "vol_5","vol_10","vol_20","vol_ratio",
    "price_vs_sma5","price_vs_sma20","price_vs_sma50","sma5_vs_sma20",
    "rsi_norm","macd_norm","bb_pos","hl_range","hl_range_5",
    "vol_ratio_v","mom_5","mom_10","mom_20","dow",
]


def train_lgbm(df_feat):
    X, y = df_feat[FEAT_COLS].values, df_feat["target"].values
    tscv = TimeSeriesSplit(n_splits=5)
    maes, oof = [], np.zeros(len(y))
    print("[ML] Training LightGBM (5-fold time-series CV) ...")
    for fold, (tr, va) in enumerate(tscv.split(X)):
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(X[tr], y[tr], eval_set=[(X[va],y[va])],
              callbacks=[lgb.early_stopping(50,verbose=False),
                         lgb.log_evaluation(-1)])
        oof[va] = m.predict(X[va])
        mae = mean_absolute_error(y[va], oof[va])
        maes.append(mae)
        print(f"    Fold {fold+1}: MAE={mae:.6f}  best_iter={m.best_iteration_}")
    dir_acc = np.mean(np.sign(oof[oof!=0]) == np.sign(y[oof!=0])) * 100
    print(f"\n[ML] CV MAE={np.mean(maes):.6f}  Dir Acc={dir_acc:.1f}%\n")
    final = lgb.LGBMRegressor(**LGB_PARAMS)
    final.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
    return final, np.mean(maes), dir_acc


def ml_base_params(model, df_feat):
    """ML drift + EWMA vol as baseline (informational only — calibration overrides both)."""
    row     = df_feat[FEAT_COLS].iloc[-1].values.copy()
    mu_ml   = float(model.predict(row.reshape(1,-1))[0])
    log_r   = df_feat["log_ret"].values
    var_ewma = log_r[-60:].var()
    var_ewma = 0.06 * log_r[-1]**2 + 0.94 * var_ewma
    sigma_ml = float(np.sqrt(var_ewma))
    return mu_ml, sigma_ml


# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────

def plot_all(S0, results, feat_imp, base_mu, base_sigma):
    DARK, PANEL = "#0d1117", "#161b22"
    CYAN, GREEN, RED = "#58c4dd", "#3fb950", "#f85149"
    YELLOW, PURPLE = "#e3b341", "#bb86fc"
    TEXT, MUTED = "#c9d1d9", "#484f58"

    days  = list(results.keys())
    ncols = 3
    nrows = (len(days) + ncols - 1) // ncols

    fig = plt.figure(figsize=(22, 5*nrows + 6), facecolor=DARK)
    outer = gridspec.GridSpec(nrows + 1, 1, figure=fig, hspace=0.55)
    top   = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.35)

    # ── Top row: feature importance, KL summary, param evolution ──
    ax_feat = fig.add_subplot(top[0])
    ax_kl   = fig.add_subplot(top[1])
    ax_par  = fig.add_subplot(top[2])

    for ax in [ax_feat, ax_kl, ax_par]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(MUTED)
        ax.title.set_color("white"); ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

    # Feature importance
    top10 = feat_imp.head(10)
    bars  = ax_feat.barh(top10.index[::-1], top10.values[::-1], color=CYAN, alpha=0.8)
    ax_feat.set_title("LightGBM feature importance", pad=5)
    for bar, val in zip(bars, top10.values[::-1]):
        ax_feat.text(bar.get_width()+top10.max()*0.01, bar.get_y()+bar.get_height()/2,
                     f"{val:.0f}", va="center", color=TEXT, fontsize=7)

    # KL per day
    kls   = [results[d]["kl"] for d in days]
    cols  = [GREEN if k<0.005 else YELLOW if k<0.02 else RED for k in kls]
    ax_kl.bar(days, kls, color=cols, alpha=0.85)
    ax_kl.axhline(0.005, color=GREEN,  lw=1, ls="--", alpha=0.7, label="Excellent <0.005")
    ax_kl.axhline(0.020, color=YELLOW, lw=1, ls="--", alpha=0.7, label="Good <0.020")
    ax_kl.set_title("KL-divergence per day", pad=5); ax_kl.set_ylabel("KL")
    ax_kl.tick_params(axis='x', rotation=30)
    ax_kl.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT, fontsize=7)

    # Calibrated mu / sigma evolution
    mus    = [results[d]["mu_r"]*100   for d in days]
    sigmas = [results[d]["sigma_r"]*100 for d in days]
    ax2    = ax_par.twinx()
    ax_par.plot(days, mus,    "o-", color=GREEN,  lw=2, ms=6, label="mu_r (%)")
    ax2.plot(   days, sigmas, "s-", color=PURPLE, lw=2, ms=6, label="sigma_r (%)")
    ax_par.axhline(base_mu*100,    color=GREEN,  lw=1, ls=":", alpha=0.5, label="ML drift")
    ax2.axhline(   base_sigma*100, color=PURPLE, lw=1, ls=":", alpha=0.5, label="EWMA vol")
    ax_par.set_title("Calibrated drift & vol per day", pad=5)
    ax_par.set_ylabel("Daily drift (%)", color=GREEN)
    ax2.set_ylabel("Daily vol (%)", color=PURPLE)
    ax_par.tick_params(colors=TEXT, labelsize=8); ax2.tick_params(colors=PURPLE, labelsize=7)
    ax_par.tick_params(axis='x', rotation=30)
    lines = ax_par.get_lines() + ax2.get_lines()
    ax_par.legend(lines, [l.get_label() for l in lines],
                  facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT, fontsize=7, loc="lower left")

    # ── Per-day bracket charts ──
    brac_gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols,
                subplot_spec=outer[1:], hspace=0.65, wspace=0.35)

    for di, day in enumerate(days):
        row, col = di // ncols, di % ncols
        ax = fig.add_subplot(brac_gs[row, col])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(MUTED)
        ax.title.set_color("white"); ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

        res  = results[day]
        lbls = [b[0] for b in res["brackets"]]
        tgt  = res["target"]
        mdl  = res["model_probs"]
        x    = np.arange(len(lbls)); w = 0.38

        ax.bar(x-w/2, tgt*100, w, color=YELLOW, alpha=0.85, label="Polymarket")
        ax.bar(x+w/2, mdl*100, w, color=GREEN,  alpha=0.80, label="Calibrated model")

        # Residual dot above each pair
        for xi, (t, m) in enumerate(zip(tgt, mdl)):
            diff  = (m - t) * 100
            dcol  = GREEN if abs(diff)<=3 else (YELLOW if abs(diff)<=6 else RED)
            ax.plot(xi, max(t,m)*100 + 1.2, "o", color=dcol, ms=4, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=40, ha="right", fontsize=6)
        grade = ("EXCELLENT" if res["kl"]<0.005 else
                 "GOOD"      if res["kl"]<0.020 else
                 "FAIR"      if res["kl"]<0.100 else "POOR")
        ax.set_title(f"{day}  KL={res['kl']:.5f}  [{grade}]", pad=4, fontsize=9)
        ax.set_ylabel("Probability (%)", fontsize=7)
        ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT, fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    fig.suptitle(
        f"BTC/USD — Normal Distribution + Polymarket Calibration  |  S0=${S0:,.0f}  |  "
        f"{PATHS:,} paths  |  KL: green<0.005 | yellow<0.020 | red>0.020",
        color=TEXT, fontsize=13, y=1.005, fontweight="bold")

    out = "btc_monte_carlo_normal.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"\n[PLOT] Saved → {out}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "="*65)
    print("  BTC MONTE CARLO — NORMAL + POLYMARKET CALIBRATION")
    print("="*65 + "\n")

    # 1. Data + ML
    df      = fetch_data(TICKER, LOOKBACK)
    df_feat = make_features(df)
    S0      = float(df["close"].iloc[-1])
    print(f"[DATA] S0 = ${S0:,.2f}\n")

    model, cv_mae, dir_acc = train_lgbm(df_feat)
    base_mu, base_sigma    = ml_base_params(model, df_feat)

    feat_imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    print(f"[ML]  base drift (ML)  = {base_mu*100:+.4f}%")
    print(f"[ML]  base vol  (EWMA) = {base_sigma*100:.4f}%")
    print(f"[ML]  Note: calibration will override these with Polymarket-fitted values\n")

    # 2. Calibrate each day against Polymarket
    results = {}

    for day, brackets in POLYMARKET.items():
        n_days = list(POLYMARKET.keys()).index(day) + 1

        print(f"\n{'='*60}")
        print(f"  {day}  ({n_days}-day horizon from S0)")
        print(f"{'='*60}")

        mu_r, sigma_r, kl, model_probs, target = calibrate_normal(
            S0, brackets, verbose=True)

        print_bracket_report(brackets, target, model_probs,
                              label=f"{day} — Normal calibrated", kl=kl)

        # Run MC with calibrated params
        sims = run_mc(S0, mu_r, sigma_r, n_days, PATHS)
        final = sims[-1]
        p5, p25, p50, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])
        prob_up = (final > S0).mean() * 100

        print(f"[SIM] {PATHS:,} paths | "
              f"P5=${p5:,.0f}  P25=${p25:,.0f}  Median=${p50:,.0f}  "
              f"P75=${p75:,.0f}  P95=${p95:,.0f}  P(up)={prob_up:.1f}%\n")

        results[day] = {
            "brackets":    brackets,
            "target":      target,
            "model_probs": model_probs,
            "mu_r":        mu_r,
            "sigma_r":     sigma_r,
            "kl":          kl,
            "sims":        sims,
        }

    # 3. Summary table
    print("\n" + "="*70)
    print("  CALIBRATION SUMMARY")
    print("="*70)
    print(f"  {'Day':<8}  {'mu_r':>9}  {'sigma_r':>9}  {'Ann.Vol':>8}  "
          f"{'KL':>12}  {'Grade':>10}")
    print(f"  {'-'*68}")
    for day, res in results.items():
        mu_r, sigma_r, kl = res["mu_r"], res["sigma_r"], res["kl"]
        ann_vol = sigma_r * np.sqrt(365) * 100
        grade   = ("EXCELLENT" if kl<0.005 else
                   "GOOD"      if kl<0.020 else
                   "FAIR"      if kl<0.100 else "POOR")
        print(f"  {day:<8}  {mu_r*100:>+8.4f}%  {sigma_r*100:>8.4f}%  "
              f"{ann_vol:>7.1f}%  {kl:>12.8f}  {grade:>10}")
    print(f"\n  ML baseline: drift={base_mu*100:+.4f}%  vol={base_sigma*100:.4f}%  "
          f"CV MAE={cv_mae:.6f}  Dir Acc={dir_acc:.1f}%")
    print("="*70 + "\n")

    # 4. Plot
    plot_all(S0, results, feat_imp, base_mu, base_sigma)

    return results, model


if __name__ == "__main__":
    results, model = run_pipeline()