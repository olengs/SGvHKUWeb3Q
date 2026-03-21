import requests
import json
import re
from datetime import datetime, timedelta

EVENT_SLUG_URL = f"https://gamma-api.polymarket.com/events/slug"
ORDER_BOOK_URL = "https://clob.polymarket.com/book"
MAX_SPREAD = 0.07

class PolyMarket():
    def __init__(self):
        pass

    def highestPrice(self, data):
        return max(data, key=lambda x: x["price"])["price"]

    def lowestPrice(self, data):
        return min(data, key=lambda x: x["price"])["price"]

    def get_max_bid_min_ask(self, token_id):
        '''
            Takes in token_id and returns a tuple
            Output: (Highest bid, lowest ask)
        '''
        data = requests.get(f"{ORDER_BOOK_URL}?token_id={token_id}").json()
        if not data or "error" in data or "bids" not in data or "asks" not in data:
            return None
        if not data["bids"] or not data["asks"]:
            return None
        return {"bid": float(self.highestPrice(data["bids"])), "ask": float(self.lowestPrice(data["asks"]))}
    
    def get_market_data(self, slug):
        try:
            slug = "-".join(slug.lower().split())
            data = requests.get(f"{EVENT_SLUG_URL}/{slug}", timeout=20).json()
            market = data["markets"]
            ret = []
            for i, predictions in enumerate(market):
                yes_id = json.loads(predictions["clobTokenIds"])[0]
                p = float(json.loads(predictions["outcomePrices"])[0])
                question = predictions["question"]
                yes_spread = self.get_max_bid_min_ask(yes_id)
                if not yes_spread:
                    continue
                if abs(yes_spread["bid"] - yes_spread["ask"]) <= MAX_SPREAD:
                    ret.append({"Question": question, "Yes": yes_spread, "p": p})
            return ret
        except:
            return None
    
    @staticmethod
    def print_market_data(data):
        for data in sorted(data, key=lambda x: x["Question"]):
            print("Question:", data["Question"], "\nYes:", data["Yes"], "\nProbability:", data["p"])


def extract_bracket_label(question: str) -> str:
    q = question.lower()
    nums = [
        int(n.replace(",", ""))
        for n in re.findall(r"[\d,]+", q)
        if int(n.replace(",", "")) >= 1000
    ]

    if not nums:
        return question
    if "greater than" in q or "above" in q:
        return f">${nums[0] // 1000}k"
    if "less than" in q or "below" in q or "dip" in q:
        return f"<{nums[0] // 1000}k"
    if len(nums) >= 2:
        lo, hi = sorted(nums[:2])
        return f"{lo // 1000}-{hi // 1000}k"
    return f"{nums[0] // 1000}k"

def fetch_polymarket_bands(market: PolyMarket) -> tuple[dict, dict]:
    market = PolyMarket()
    all_results = {}
    bands = {}
    date = datetime.now()
    counter = 0

    while counter < 3:
        label = f"{date.strftime("%b")} {int(date.strftime("%d"))}"
        slug = f"bitcoin-price-on-{date.strftime("%B").lower()}-{int(date.strftime("%d"))}"

        try:
            data = market.get_market_data(slug)
            if not data:
                print("  [!] No liquid markets found (slug may not exist yet or spreads too wide)")
                all_results[label] = []
                counter += 1
                date += timedelta(1)
                continue

            print(f"\n{'=' * 50}")
            print(f"  Bitcoin Price — {label}  [{slug}]")
            print(f"{'=' * 50}")

            date += timedelta(1)

            PolyMarket.print_market_data(data)
            all_results[label] = data

            bands[label] = [
                {
                    "b": extract_bracket_label(item["Question"]),
                    "p": round(item["p"], 4),
                    "bid": item["Yes"]["bid"],
                    "ask": item["Yes"]["ask"],
                }
                for item in sorted(data, key=lambda x: x["Question"])
            ]

        except Exception as e:
            print(f"  [!] Error fetching {label}: {e}")
            all_results[label] = []

    print(f"\n{'=' * 50}")
    print("  Top bracket per day")
    print(f"{'=' * 50}")

    for label, data in all_results.items():
        if not data:
            print(f"  {label}: no data")
            continue

        top = max(data, key=lambda x: x["p"])
        lbl = extract_bracket_label(top["Question"])
        print(
            f"  {label}: {lbl}  @  {float(top['p']) * 100:.1f}%"
            f"  |  bid={top['Yes']['bid']:.3f}  ask={top['Yes']['ask']:.3f}"
        )

    return all_results, bands