from polymarket import PolyMarket, fetch_polymarket_bands
import os
import json
import dotenv
from detection import Detection
import time
import btc_monte_carlo
from roostoo import Roostoo
from mylogger import MyLogger
from trade_trigger import evaluate
from datetime import datetime

dotenv.load_dotenv()

MODE = os.getenv("MODE")
API_KEY = os.getenv(f"{MODE}_API_KEY")
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET")

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

def buy_from_roostoo(broker:Roostoo, coin, amount):
    last_price = float(broker.get_ticker_last_price(coin)) - 1
    quantity = round(amount / last_price, 5)
    roostoo.place_order(coin, "BUY", quantity, last_price)
    return quantity, last_price

def sell_from_roostoo(broker:Roostoo, coin):
    last_price = float(broker.get_ticker_last_price(coin)) + 1
    holdings = float(broker.get_balance()["SpotWallet"]["BTC"]["Free"]) * 0.5
    holdings = round(holdings, 5)
    roostoo.place_order(coin, "SELL", holdings, last_price)
    return holdings, last_price


if __name__ == "__main__":
    coin = "BTC/USD"
    market_detector = Detection(coin, 1000)
    polymarket = PolyMarket()
    logger = MyLogger("log.txt")
    roostoo = Roostoo(API_KEY, SECRET_KEY)

    while True:
        execeed_thresh, diff = market_detector.update()
        prev_order = None
        if not execeed_thresh:
            logger.log(f"price did not execeed threshold: at last price: {market_detector.get_ticker_last_price()}, waiting for next hour...")
            time.sleep(2 * 60 * 60) #check every 2 hours
            continue

        today_bands = fetch_today_bands()
        if today_bands is None:
            print("[MAIN] No actionable Polymarket data — skipping cycle")
            time.sleep(60 * 60)
            continue

        S0 = market_detector.get_ticker_last_price()
        print(f"[MAIN] Spot: ${S0:,.2f}")

        # ── Core signal ──────────────────────────────────────────────────────
        signal, detail = evaluate(today_bands, position=prev_order, S0=S0)

        logger.log(f"Signal details: {detail}")

        #Warning triggered and went down
        logger.log(f"Monte carlo returned results: {signal}, at last price: {market_detector.get_ticker_last_price()}")
        if signal == 0:
            continue

        if signal < 0:
            quantity, amt = sell_from_roostoo(roostoo, coin)
            logger.log(f"Sold {coin} of quantity: {quantity} at price: {amt}")
            prev_order = None
            pass

        else: #Warning triggered and went up
            quantity, amt = buy_from_roostoo(roostoo, coin, 50000)
            logger.log(f"Bought {coin} of quantity: {quantity} at price: {amt}")
            prev_order = {"entry_price": amt, "qty": quantity}
