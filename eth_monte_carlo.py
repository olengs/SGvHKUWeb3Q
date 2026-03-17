"""
ETH Monte Carlo Simulation  —  ML Edition
==========================================
Architecture:
    1. Fetch 3Y ETH-USD history via yfinance
    2. Engineer technical features (RSI, MACD, Bollinger, vol ratios, etc.)
    3. Train LightGBM to predict next-day log-return (drift)
    4. Use LightGBM's day-by-day drift forecast to bias GBM paths
       → Each future day gets its own ML-predicted mu instead of flat historical mean
    5. Uncertainty = GARCH-style rolling volatility (not flat sigma)
    6. Output: day-by-day forecast table + 4-panel chart

Key insight:
    Vanilla GBM uses one fixed mu for all 15 days.
    This model uses ML to ask "given today's signals, what direction is likely?"
    and plugs that per-day answer into the Monte Carlo engine.

Requirements:
    pip install yfinance lightgbm scikit-learn pandas numpy matplotlib scipy

Usage:
    python eth_monte_carlo_ml.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TICKER        = "ETH-USD"
LOOKBACK      = 3          # years of history
SIM_DAYS      = 15
PATHS         = 10_000
SEED          = 42

# LightGBM hyperparameters (tuned for crypto time-series)
LGB_PARAMS = {
    "objective"        : "regression",
    "metric"           : "mae",
    "n_estimators"     : 500,
    "learning_rate"    : 0.03,
    "num_leaves"       : 31,
    "max_depth"        : 5,
    "feature_fraction" : 0.8,
    "bagging_fraction" : 0.8,
    "bagging_freq"     : 5,
    "min_child_samples": 20,
    "lambda_l1"        : 0.1,
    "lambda_l2"        : 0.1,
    "verbose"          : -1,
    "random_state"     : SEED,
}


# ─────────────────────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────────────────────
def fetch_data(ticker: str, lookback_years: int) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=lookback_years * 365)
    print(f"[DATA] Fetching {ticker} ({start.date()} → {end.date()}) ...")
    raw = yf.download(ticker, start=start, end=end, progress=False)
    df  = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.dropna(inplace=True)
    print(f"[DATA] {len(df)} trading days loaded.\n")
    return df


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
#    All features use only past data — no lookahead leakage
# ─────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # ── Returns ───────────────────────────────────────────────
    d["log_ret"]      = np.log(d["close"] / d["close"].shift(1))
    d["log_ret_2"]    = np.log(d["close"] / d["close"].shift(2))
    d["log_ret_5"]    = np.log(d["close"] / d["close"].shift(5))
    d["log_ret_10"]   = np.log(d["close"] / d["close"].shift(10))
    d["log_ret_20"]   = np.log(d["close"] / d["close"].shift(20))

    # ── Volatility ─────────────────────────────────────────────
    d["vol_5"]        = d["log_ret"].rolling(5).std()
    d["vol_10"]       = d["log_ret"].rolling(10).std()
    d["vol_20"]       = d["log_ret"].rolling(20).std()
    d["vol_ratio"]    = d["vol_5"] / (d["vol_20"] + 1e-9)   # vol regime

    # ── Moving averages & crossovers ──────────────────────────
    d["sma_5"]        = d["close"].rolling(5).mean()
    d["sma_20"]       = d["close"].rolling(20).mean()
    d["sma_50"]       = d["close"].rolling(50).mean()
    d["price_vs_sma5"]  = d["close"] / (d["sma_5"]  + 1e-9) - 1
    d["price_vs_sma20"] = d["close"] / (d["sma_20"] + 1e-9) - 1
    d["price_vs_sma50"] = d["close"] / (d["sma_50"] + 1e-9) - 1
    d["sma5_vs_sma20"]  = d["sma_5"]  / (d["sma_20"] + 1e-9) - 1

    # ── RSI (14) ───────────────────────────────────────────────
    delta   = d["close"].diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / (loss + 1e-9)
    d["rsi_14"] = 100 - (100 / (1 + rs))
    d["rsi_norm"] = (d["rsi_14"] - 50) / 50   # normalise to [-1, 1]

    # ── MACD ──────────────────────────────────────────────────
    ema12       = d["close"].ewm(span=12, adjust=False).mean()
    ema26       = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]   = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]
    d["macd_norm"]   = d["macd_hist"] / (d["close"] + 1e-9)

    # ── Bollinger Bands ────────────────────────────────────────
    bb_mid      = d["close"].rolling(20).mean()
    bb_std      = d["close"].rolling(20).std()
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_pos"]   = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)  # 0=bottom, 1=top

    # ── High-Low range ─────────────────────────────────────────
    d["hl_range"]   = (d["high"] - d["low"]) / (d["close"] + 1e-9)
    d["hl_range_5"] = d["hl_range"].rolling(5).mean()

    # ── Volume signals ─────────────────────────────────────────
    d["vol_sma20"]  = d["volume"].rolling(20).mean()
    d["vol_ratio_v"]= d["volume"] / (d["vol_sma20"] + 1e-9)

    # ── Momentum ───────────────────────────────────────────────
    d["mom_5"]  = d["close"] / (d["close"].shift(5)  + 1e-9) - 1
    d["mom_10"] = d["close"] / (d["close"].shift(10) + 1e-9) - 1
    d["mom_20"] = d["close"] / (d["close"].shift(20) + 1e-9) - 1

    # ── Day-of-week effect ─────────────────────────────────────
    d["dow"] = d.index.dayofweek.astype(float) / 6.0

    # ── TARGET: next-day log return ────────────────────────────
    d["target"] = d["log_ret"].shift(-1)   # predict tomorrow's return

    return d.dropna()


FEATURE_COLS = [
    "log_ret", "log_ret_2", "log_ret_5", "log_ret_10", "log_ret_20",
    "vol_5", "vol_10", "vol_20", "vol_ratio",
    "price_vs_sma5", "price_vs_sma20", "price_vs_sma50", "sma5_vs_sma20",
    "rsi_norm", "macd_norm", "bb_pos",
    "hl_range", "hl_range_5",
    "vol_ratio_v", "mom_5", "mom_10", "mom_20", "dow",
]


# ─────────────────────────────────────────────────────────────
# 3. TRAIN LIGHTGBM  —  Walk-forward time-series CV
# ─────────────────────────────────────────────────────────────
def train_lgbm(df_feat: pd.DataFrame):
    X = df_feat[FEATURE_COLS].values
    y = df_feat["target"].values

    # Time-series split — never use future data for training
    tscv      = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(y))
    val_maes  = []

    print("[ML] Training LightGBM with 5-fold time-series CV ...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        oof_preds[val_idx] = model.predict(X_val)
        mae = mean_absolute_error(y_val, oof_preds[val_idx])
        val_maes.append(mae)
        print(f"    Fold {fold+1}: MAE = {mae:.6f}  |  best iter = {model.best_iteration_}")

    mean_mae = np.mean(val_maes)
    # Directional accuracy on OOF
    dir_acc  = np.mean(np.sign(oof_preds[oof_preds != 0]) ==
                       np.sign(y[oof_preds != 0])) * 100

    print(f"\n[ML] CV Mean MAE      : {mean_mae:.6f}")
    print(f"[ML] OOF Dir Accuracy : {dir_acc:.1f}%  (random = 50%)\n")

    # Retrain on ALL data for final forecasting
    print("[ML] Retraining on full dataset ...")
    final_model = lgb.LGBMRegressor(**LGB_PARAMS)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])

    return final_model, oof_preds, mean_mae, dir_acc


# ─────────────────────────────────────────────────────────────
# 4. FORECAST ML DRIFT FOR NEXT 15 DAYS
#    Day 1: use real last features
#    Day 2+: auto-regress on simulated median price
# ─────────────────────────────────────────────────────────────
def forecast_ml_drift(model, df_feat: pd.DataFrame, sim_days: int) -> np.ndarray:
    """
    Returns array of shape (sim_days,) — ML-predicted drift per day.
    Day 1 uses real latest features.
    Days 2-15 use a simple auto-regressive approximation.
    """
    ml_drifts = np.zeros(sim_days)
    last_row   = df_feat[FEATURE_COLS].iloc[-1].values.copy()

    for day in range(sim_days):
        pred = model.predict(last_row.reshape(1, -1))[0]
        ml_drifts[day] = pred

        # Shift lag features forward (simplified auto-regression)
        # In practice: log_ret[t] becomes log_ret_2[t+1] etc.
        last_row[0] = pred                                    # log_ret = predicted
        last_row[1] = last_row[0]                             # log_ret_2
        last_row[2] = (last_row[2] * 4 + pred) / 5           # log_ret_5 rolling approx
        last_row[3] = (last_row[3] * 9 + pred) / 10          # log_ret_10
        last_row[4] = (last_row[4] * 19 + pred) / 20         # log_ret_20
        # vol: update with rolling std approximation
        last_row[5] = np.sqrt((last_row[5]**2 * 4 + pred**2) / 5)   # vol_5
        last_row[6] = np.sqrt((last_row[6]**2 * 9 + pred**2) / 10)
        last_row[7] = np.sqrt((last_row[7]**2 * 19 + pred**2) / 20)

    print(f"[ML] 15-day drift forecast (annualised): "
          f"{ml_drifts.mean()*365*100:.1f}%\n")
    return ml_drifts


# ─────────────────────────────────────────────────────────────
# 5. ROLLING VOLATILITY (GARCH-lite)
#    Use exponentially weighted vol for each forecast day
# ─────────────────────────────────────────────────────────────
def forecast_vol(df_feat: pd.DataFrame, sim_days: int, alpha: float = 0.06) -> np.ndarray:
    """
    EWMA volatility forecast (simple GARCH-lite).
    sigma_t^2 = alpha * r_{t-1}^2 + (1-alpha) * sigma_{t-1}^2
    """
    log_rets  = df_feat["log_ret"].values
    var_hist  = log_rets[-60:].var()   # initialise from recent 60-day variance

    vols = np.zeros(sim_days)
    var  = var_hist
    for day in range(sim_days):
        last_r2 = log_rets[-1]**2 if day == 0 else vols[day-1]**2
        var     = alpha * last_r2 + (1 - alpha) * var
        vols[day] = np.sqrt(var)

    return vols


# ─────────────────────────────────────────────────────────────
# 6. MONTE CARLO  —  ML drift + EWMA vol
# ─────────────────────────────────────────────────────────────
def run_monte_carlo_ml(
    S0: float,
    ml_drifts: np.ndarray,     # shape (sim_days,)  — ML predicted mu per day
    ewma_vols: np.ndarray,     # shape (sim_days,)  — EWMA sigma per day
    sim_days: int,
    paths: int,
    seed: int = SEED,
) -> np.ndarray:
    """
    GBM where each day t uses:
        mu_t    = ml_drifts[t]   (from LightGBM)
        sigma_t = ewma_vols[t]   (from EWMA)
    Returns simulations of shape (sim_days+1, paths).
    """
    rng   = np.random.default_rng(seed)
    sims  = np.empty((sim_days + 1, paths))
    sims[0] = S0

    for t in range(sim_days):
        mu_t    = ml_drifts[t]
        sig_t   = ewma_vols[t]
        drift   = mu_t - 0.5 * sig_t**2
        Z       = rng.standard_normal(paths)
        sims[t+1] = sims[t] * np.exp(drift + sig_t * Z)

    return sims


# ─────────────────────────────────────────────────────────────
# 7. PRINT DAY-BY-DAY TABLE
# ─────────────────────────────────────────────────────────────
def print_daily_forecast(sims: np.ndarray, S0: float,
                         ml_drifts: np.ndarray, ewma_vols: np.ndarray,
                         start_date: datetime):
    W = 128
    print("\n" + "=" * W)
    print(f"  DAY-BY-DAY ETH FORECAST  |  ML + Monte Carlo  |  "
          f"Start: ${S0:,.2f}  |  {PATHS:,} paths")
    print("=" * W)
    print(f"  {'Day':<4} {'Date':<12} {'ML Drift':>9} {'EWMA Vol':>9} "
          f"{'Mean':>10} {'Median':>10} {'5th':>10} {'25th':>10} "
          f"{'75th':>10} {'95th':>10}  {'VaR95$':>8}  {'P(up)':>6}")
    print(f"  {'-'*(W-2)}")

    for day in range(1, sims.shape[0]):
        prices   = sims[day, :]
        date_str = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")
        mean     = prices.mean()
        median   = np.median(prices)
        p5, p25, p75, p95 = np.percentile(prices, [5, 25, 75, 95])
        var95    = S0 - p5
        cvar95   = S0 - prices[prices <= p5].mean()
        prob_up  = (prices > S0).mean() * 100
        drift_pct = ml_drifts[day-1] * 100
        vol_pct   = ewma_vols[day-1] * 100

        print(
            f"  {day:<4} {date_str:<12} {drift_pct:>+8.3f}% {vol_pct:>8.3f}%  "
            f"${mean:>9,.2f} ${median:>9,.2f} "
            f"${p5:>9,.2f} ${p25:>9,.2f} ${p75:>9,.2f} ${p95:>9,.2f}  "
            f"${var95:>7,.2f}  {prob_up:>5.1f}%"
        )

    print("=" * W)
    print("  ML Drift = LightGBM predicted daily log-return  |  "
          "EWMA Vol = exponentially-weighted daily volatility  |  "
          "VaR95 = max dollar loss at 95% confidence")
    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE SUMMARY
# ─────────────────────────────────────────────────────────────
def print_feature_importance(model):
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)
    print("[ML] TOP 10 FEATURE IMPORTANCES")
    print("=" * 40)
    for feat, val in imp.head(10).items():
        bar = "█" * int(val / imp.max() * 20)
        print(f"  {feat:<22} {bar:<20} {val:.0f}")
    print("=" * 40 + "\n")
    return imp


# ─────────────────────────────────────────────────────────────
# 9. POLYMARKET SIGNALS
# ─────────────────────────────────────────────────────────────
def generate_signals(sims: np.ndarray, S0: float):
    final = sims[-1, :]
    levels = {
        f"ETH > ${S0*1.05:,.0f} (+5%)" : np.mean(final > S0 * 1.05),
        f"ETH > ${S0*1.10:,.0f} (+10%)": np.mean(final > S0 * 1.10),
        f"ETH > ${S0*1.20:,.0f} (+20%)": np.mean(final > S0 * 1.20),
        f"ETH < ${S0*0.95:,.0f} (-5%)" : np.mean(final < S0 * 0.95),
        f"ETH < ${S0*0.90:,.0f} (-10%)": np.mean(final < S0 * 0.90),
        f"ETH < ${S0*0.80:,.0f} (-20%)": np.mean(final < S0 * 0.80),
    }
    print("=" * 55)
    print("  POLYMARKET SIGNALS  (ML-enhanced probabilities)")
    print("=" * 55)
    for market, prob in levels.items():
        implied = 0.50
        edge    = prob - implied
        flag    = "[BET ]" if abs(edge) > 0.05 else "[SKIP]"
        direct  = "YES" if prob > 0.50 else "NO"
        print(f"  {flag} {market}")
        print(f"         Model: {prob:.2%}  |  Implied: {implied:.2%}  |  "
              f"Edge: {edge:+.2%}  →  {direct}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────
# 10. PLOT — 4-panel chart
# ─────────────────────────────────────────────────────────────
def plot_results(sims, df, df_feat, ml_drifts, ewma_vols,
                 feat_imp, start_date, S0):
    DARK, PANEL = "#0d1117", "#161b22"
    CYAN, GREEN, RED = "#58c4dd", "#3fb950", "#f85149"
    YELLOW, PURPLE = "#e3b341", "#bb86fc"
    TEXT, MUTED = "#c9d1d9", "#484f58"

    fig = plt.figure(figsize=(18, 13), facecolor=DARK)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, :])   # full-width fan chart
    ax2 = fig.add_subplot(gs[1, 0])   # feature importance
    ax3 = fig.add_subplot(gs[1, 1])   # ML drift + vol overlay

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor(MUTED)

    # ── Panel 1: Fan chart ─────────────────────────────────────
    days_axis    = np.arange(sims.shape[0])
    future_dates = [start_date + timedelta(days=i) for i in range(sims.shape[0])]

    # Faint path sample
    for i in np.random.choice(PATHS, 200, replace=False):
        col = GREEN if sims[-1, i] >= S0 else RED
        ax1.plot(days_axis, sims[:, i], alpha=0.04, lw=0.4, color=col)

    # Confidence bands
    p5  = np.percentile(sims, 5,  axis=1)
    p25 = np.percentile(sims, 25, axis=1)
    p50 = np.percentile(sims, 50, axis=1)
    p75 = np.percentile(sims, 75, axis=1)
    p95 = np.percentile(sims, 95, axis=1)

    ax1.fill_between(days_axis, p5,  p95,  alpha=0.10, color=RED,  label="5–95th pct")
    ax1.fill_between(days_axis, p25, p75,  alpha=0.22, color=CYAN, label="25–75th pct")
    ax1.plot(days_axis, p50,  color=GREEN,  lw=2.2, label="Median (ML-biased)")
    ax1.plot(days_axis, p5,   color=RED,    lw=1.0, ls="--", alpha=0.7)
    ax1.plot(days_axis, p95,  color=GREEN,  lw=1.0, ls="--", alpha=0.7)
    ax1.axhline(S0, color=YELLOW, lw=1.2, ls=":", alpha=0.8, label=f"Entry ${S0:,.0f}")

    ax1.set_title(
        f"ETH Monte Carlo  —  ML (LightGBM) drift + EWMA vol  |  "
        f"{PATHS:,} paths  |  {SIM_DAYS}-day forecast",
        color=TEXT, fontsize=12, pad=10
    )
    ax1.set_xlabel("Forecast Day", color=TEXT)
    ax1.set_ylabel("Price (USD)", color=TEXT)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT, fontsize=8, loc="upper left")

    # ── Panel 2: Feature importance ────────────────────────────
    top10 = feat_imp.head(10)
    bars  = ax2.barh(top10.index[::-1], top10.values[::-1],
                     color=CYAN, alpha=0.80)
    ax2.set_title("LightGBM Feature Importance\n(top 10 drivers of ETH drift)", pad=8)
    ax2.set_xlabel("Importance Score")
    for bar, val in zip(bars, top10.values[::-1]):
        ax2.text(bar.get_width() + top10.max()*0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f}", va="center", color=TEXT, fontsize=7)

    # ── Panel 3: ML drift & EWMA vol forecast ─────────────────
    days = np.arange(1, SIM_DAYS + 1)
    ax3.bar(days, ml_drifts * 100, color=[GREEN if v >= 0 else RED for v in ml_drifts],
            alpha=0.75, label="ML daily drift (%)")
    ax3.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax3.set_ylabel("ML Predicted Drift (%)", color=TEXT)
    ax3.set_xlabel("Forecast Day")
    ax3.set_title("LightGBM Drift & EWMA Volatility per Day")

    ax3b = ax3.twinx()
    ax3b.plot(days, ewma_vols * 100, color=PURPLE, lw=2, marker="o",
              ms=4, label="EWMA vol (%)")
    ax3b.set_ylabel("EWMA Daily Vol (%)", color=PURPLE, fontsize=8)
    ax3b.tick_params(colors=PURPLE, labelsize=7)
    ax3b.spines["right"].set_edgecolor(PURPLE)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2,
               facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)

    fig.suptitle(
        f"ETH/USD  |  LightGBM + GBM Monte Carlo  |  Lookback: {LOOKBACK}Y  |  "
        f"Paths: {PATHS:,}  |  Horizon: {SIM_DAYS}d",
        color=TEXT, fontsize=13, y=0.99, fontweight="bold"
    )

    out = "eth_monte_carlo_ml.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"[PLOT] Saved → {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline():
    print("\n" + "=" * 60)
    print("  ETH MONTE CARLO  —  ML EDITION  (LightGBM + GBM)")
    print("=" * 60 + "\n")

    # 1. Data
    df      = fetch_data(TICKER, LOOKBACK)
    df_feat = make_features(df)

    print(f"[FEAT] {len(FEATURE_COLS)} features engineered on {len(df_feat)} rows.\n")

    # 2. Train LightGBM
    model, oof_preds, cv_mae, dir_acc = train_lgbm(df_feat)

    # 3. Feature importance
    feat_imp = print_feature_importance(model)

    # 4. Forecast drift & vol for next 15 days
    ml_drifts = forecast_ml_drift(model, df_feat, SIM_DAYS)
    ewma_vols = forecast_vol(df_feat, SIM_DAYS)

    # 5. Monte Carlo with ML drift
    S0         = float(df["close"].iloc[-1])
    start_date = df.index[-1].to_pydatetime()

    print(f"[SIM] Running {PATHS:,} paths × {SIM_DAYS} days from ${S0:,.2f} ...\n")
    sims = run_monte_carlo_ml(S0, ml_drifts, ewma_vols, SIM_DAYS, PATHS)

    # 6. Day-by-day table
    print_daily_forecast(sims, S0, ml_drifts, ewma_vols, start_date)

    # 7. Signals
    generate_signals(sims, S0)

    # 8. Plot
    plot_results(sims, df, df_feat, ml_drifts, ewma_vols,
                 feat_imp, start_date, S0)

    print(f"[DONE] CV MAE={cv_mae:.6f}  |  OOF Dir Acc={dir_acc:.1f}%")
    return sims, model


if __name__ == "__main__":
    sims, model = run_pipeline()