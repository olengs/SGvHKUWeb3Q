from polymarket import PolyMarket, fetch_polymarket_bands
import os
import json
import dotenv
from detection import Detection
import time

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


if __name__ == "__main__":
    market_detector = Detection("BTC/USD", 10)

    while True:
        execeed_thresh, diff = market_detector.update()
        if execeed_thresh:
            time.sleep(60 * 60)
            continue

        #Warning triggered and went down
        if diff < 0:
            pass

        else: #Warning triggered and went up
            pass