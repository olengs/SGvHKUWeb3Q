from polymarket import PolyMarket, fetch_polymarket_bands
from trade_trigger import evaluate
# from roostoo import place_order, get_ticker, get_balance
import os
import json
import dotenv
from detection import Detection
from datetime import datetime
from roostoo import Roostoo

dotenv.load_dotenv()

MODE       = os.getenv("MODE")
API_KEY    = os.getenv(f"{MODE}_API_KEY")
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET")

PAIR           = "BTC/USD"
ORDER_FRACTION = 0.10   # 10% of total portfolio value
ORDER_AMOUNT = ORDER_FRACTION * 1,000,000

# Persists across loop iterations
open_position: dict | None = None


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_today_key() -> str:
    now = datetime.now()
    return f"{now.strftime('%b')} {int(now.strftime('%d'))}"


def _parse_label(label: str):
    """Convert '72-74k' / '<62k' / '>80k' → (lo, hi) in USD."""
    import re
    nums = [int(n) * 1000 for n in re.findall(r"\d+", label)]
    if label.startswith("<"):
        return None, nums[0]
    if label.startswith(">"):
        return nums[0], None
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None


def get_spot_price() -> float:
    data = get_ticker(PAIR)
    if data and "Data" in data and PAIR in data["Data"]:
        d   = data["Data"][PAIR]
        bid = float(d.get("BidPrice", 0))
        ask = float(d.get("AskPrice", 0))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return float(d.get("LastPrice", 0))
    raise RuntimeError(f"Could not fetch price for {PAIR}")


def get_order_size_btc(S0: float) -> float:
    """10% of total portfolio (BTC + USD) converted to BTC qty."""
    bal = get_balance()
    if not bal or "Data" not in bal:
        raise RuntimeError("Could not fetch balance")
    total_usd = 0.0
    for asset, qty_str in bal["Data"].items():
        qty = float(qty_str)
        if asset == "USD":
            total_usd += qty
        elif asset == "BTC":
            total_usd += qty * S0
    if total_usd <= 0:
        raise RuntimeError("Portfolio value is zero")
    order_btc = round((total_usd * ORDER_FRACTION) / S0, 6)
    print(f"[MAIN] Portfolio ${total_usd:,.2f} → order {order_btc:.6f} BTC")
    return order_btc


# ──────────────────────────────────────────────────────────────────────────────
# MARKET ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def fetch_today_bands() -> list | None:
    """Fetch Polymarket, write JSON, return today's brackets as (label,lo,hi,prob) tuples."""
    market = PolyMarket()
    _, bands = fetch_polymarket_bands(market)

    with open("polymarket_bands.json", "w", encoding="utf-8") as f:
        json.dump(bands, f, indent=2)
    print("[MAIN] polymarket_bands.json written")

    today_key = get_today_key()
    if today_key not in bands or not bands[today_key]:
        print(f"[MAIN] No bands for today ({today_key})")
        return None

    brackets = [
        (_parse_label(b["b"])[0] and b["b"] or b["b"],   # label
         *_parse_label(b["b"]),                           # lo, hi
         float(b["p"]))
        for b in bands[today_key]
    ]
    # Rebuild cleanly as proper 4-tuples
    brackets = [
        (b["b"], *_parse_label(b["b"]), float(b["p"]))
        for b in bands[today_key]
    ]
    print(f"[MAIN] {len(brackets)} brackets loaded for {today_key}")
    return brackets


# ──────────────────────────────────────────────────────────────────────────────
# EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def execute_signal(signal: int, detail: dict, S0: float) -> None:
    """
    Translates evaluate() output into actual orders.
        1  → BUY
        0  → HOLD / SKIP (no-op)
       -1  → SELL / CUT
    """
    global open_position
    action = detail["action"]

    print(f"[MAIN] Signal={signal:+d}  Action={action}  | {detail['reason']}")
    print(f"[MAIN] Skew: bull={detail['skew']['p_bull']:.1%}  "
          f"bear={detail['skew']['p_bear']:.1%}  "
          f"ratio={detail['skew']['skew_ratio']:.2f}x")
    print(f"[MAIN] MC:   EV={detail['mc']['ev']*100:+.2f}%  "
          f"P(up)={detail['mc']['prob_up']:.1%}  "
          f"P(2h drop>$2k)={detail['mc']['p_2h_drop_2k']:.1%}  "
          f"Med=${detail['mc']['median']:,.0f}")

    if signal == 1:                         # ── BUY ──
        qty = get_order_size_btc(S0)
        order = place_order(PAIR, "BUY", qty)
        print(f"[MAIN] 🟢 BUY {qty:.6f} BTC @ ${S0:,.2f}  →  {order}")
        open_position = {"entry_price": S0, "qty": qty}

    elif signal == -1:                      # ── SELL / CUT ──
        if open_position is None:
            print("[MAIN] ⚠️  Signal -1 but no open position — skipping")
            return
        qty = open_position["qty"]
        order = place_order(PAIR, "SELL", qty)
        tag = "🔴 CUT" if action == "CUT" else "🟡 SELL"
        print(f"[MAIN] {tag} {qty:.6f} BTC @ ${S0:,.2f}  →  {order}")
        open_position = None

    else:                                   # ── HOLD / SKIP ──
        if open_position:
            entry = open_position["entry_price"]
            pnl   = (S0 - entry) / entry * 100
            print(f"[MAIN] ⏸  HOLD  entry=${entry:,.2f}  PnL={pnl:+.2f}%")
        else:
            print("[MAIN] ⏸  SKIP — conditions not met, staying flat")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    market_detector = Detection("BTC/USD", 10)
    roostoo = Roostoo(API_KEY, SECRET_KEY)

    while True:
        # if market_detector.update():
        #     continue

        # Detection triggered — run full analysis
        today_bands = fetch_today_bands()
        if today_bands is None:
            print("[MAIN] No actionable Polymarket data — skipping cycle")
            continue

        S0 = market_detector.get_ticker_last_price()
        print(f"[MAIN] Spot: ${S0:,.2f}")

        # ── Core signal ──────────────────────────────────────────────────────
        signal, detail = evaluate(today_bands, position=open_position, S0=S0)

        print(signal)
        break

        # ── Execute ──────────────────────────────────────────────────────────
        # execute_signal(signal, detail, S0)