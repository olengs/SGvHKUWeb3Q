import requests
import json

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
        slug = "-".join(slug.lower().split())
        data = requests.get(f"{EVENT_SLUG_URL}/{slug}", timeout=20).json()
        market = data["markets"]
        with open("test.json", 'w') as f:
            f.write(json.dumps(market))
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
    
    @staticmethod
    def print_market_data(data):
        for data in sorted(data, key=lambda x: x["Question"]):
            print("Question:", data["Question"], "\nYes:", data["Yes"], "\nProbability:", data["p"])