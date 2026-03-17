from roostoo import Roostoo
import os
import dotenv
dotenv.load_dotenv()
from polymarket import PolyMarket
 
MODE = os.getenv("MODE")
API_KEY = os.getenv(f"{MODE}_API_KEY")
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET")
 
DAILY_SLUGS = [
    ("March 17", "bitcoin-price-on-march-17"),
    ("March 18", "bitcoin-price-on-march-18"),
    ("March 19", "bitcoin-price-on-march-19"),
    ("March 20", "bitcoin-price-on-march-20"),
    ("March 21", "bitcoin-price-on-march-21"),
    ("March 22", "bitcoin-price-on-march-22"),
    ("March 23", "bitcoin-price-on-march-23"),
]
 
if __name__ == "__main__":
 
    broker = Roostoo(API_KEY, SECRET_KEY)
    market = PolyMarket()
 
    all_results = {}
 
    for label, slug in DAILY_SLUGS:
        print(f"\n{'='*50}")
        print(f"  Bitcoin Price — {label}")
        print(f"  Slug: {slug}")
        print(f"{'='*50}")
        try:
            data = market.get_market_data(slug)
            if not data:
                print(f"  [!] No liquid markets found for {label} (slug may not exist yet or all spreads too wide)")
                all_results[label] = []
                continue
            PolyMarket.print_market_data(data)
            all_results[label] = data
        except Exception as e:
            print(f"  [!] Error fetching {label}: {e}")
            all_results[label] = []
 
    print(f"\n{'='*50}")
    print("  Summary — brackets with highest probability per day")
    print(f"{'='*50}")
    for label, data in all_results.items():
        if not data:
            print(f"  {label}: no data")
            continue
        top = max(data, key=lambda x: x["p"])
        print(f"  {label}: {top['Question']} @ {float(top['p'])*100:.1f}%  |  bid={top['Yes']['bid']:.3f}  ask={top['Yes']['ask']:.3f}")