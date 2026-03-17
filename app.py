from polymarket import PolyMarket
import os
import json
import re
import dotenv

dotenv.load_dotenv()

MODE = os.getenv("MODE")
API_KEY = os.getenv(f"{MODE}_API_KEY")
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET")

DAILY_SLUGS = [
    ("Mar 17", "bitcoin-price-on-march-17"),
    ("Mar 18", "bitcoin-price-on-march-18"),
    ("Mar 19", "bitcoin-price-on-march-19"),
    ("Mar 20", "bitcoin-price-on-march-20"),
    ("Mar 21", "bitcoin-price-on-march-21"),
    ("Mar 22", "bitcoin-price-on-march-22"),
    ("Mar 23", "bitcoin-price-on-march-23"),
]


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


def fetch_polymarket_bands(market: PolyMarket, daily_slugs: list[tuple[str, str]]) -> tuple[dict, dict]:
    all_results = {}
    bands = {}

    for label, slug in daily_slugs:
        print(f"\n{'=' * 50}")
        print(f"  Bitcoin Price — {label}  [{slug}]")
        print(f"{'=' * 50}")

        try:
            data = market.get_market_data(slug)

            if not data:
                print("  [!] No liquid markets found (slug may not exist yet or spreads too wide)")
                all_results[label] = []
                continue

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


if __name__ == "__main__":
    market = PolyMarket()

    all_results, bands = fetch_polymarket_bands(market, DAILY_SLUGS)

    with open("polymarket_bands.json", "w", encoding="utf-8") as f:
        json.dump(bands, f, indent=2)

    print("\n[✓] polymarket_bands.json written")