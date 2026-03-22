from polymarket import PolyMarket, fetch_polymarket_bands
import os
import json
import dotenv
from detection import Detection
import time
import btc_monte_carlo
from roostoo import Roostoo
from mylogging import MyLogger

dotenv.load_dotenv()

MODE = os.getenv("MODE")
API_KEY = os.getenv(f"{MODE}_API_KEY")
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET")

def AnalyseMarket():
    #Use monte carlo to analyse
    all_results, bands = fetch_polymarket_bands()

    with open("polymarket_bands.json", "w", encoding="utf-8") as f:
        json.dump(bands, f, indent=2)

    print("\n[✓] polymarket_bands.json written")

    return True

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
    market_detector = Detection(coin, 10)
    polymarket = PolyMarket()
    logger = MyLogger("log.txt")
    roostoo = Roostoo(API_KEY, SECRET_KEY)

    while True:
        execeed_thresh, diff = market_detector.update()
        if not execeed_thresh:
            logger.log(f"price did not execeed threshold: at last price: {market_detector.get_ticker_last_price()}, waiting for next hour...")
            time.sleep(60 * 60) #check every hour

        fetch_polymarket_bands(polymarket)
        btc_monte_carlo.update_polymarket_bands()

        results, model = btc_monte_carlo.run_pipeline()
        print("ret:", results)
        with open("test_ret.txt", 'w') as f:
            f.write(json.dumps(results))

        #Warning triggered and went down
        results = 0
        logger.log(f"Monte carlo returned results: {results}, at last price: {market_detector.get_ticker_last_price()}")
        if results == 0:
            continue

        if diff < 0:
            quantity, amt = sell_from_roostoo(roostoo, coin)
            logger.log(f"Sold {coin} of quantity: {quantity} at price: {amt}")

        else: #Warning triggered and went up
            quantity, amt = buy_from_roostoo(roostoo, coin, 50000)
            logger.log(f"Bought {coin} of quantity: {quantity} at price: {amt}")
