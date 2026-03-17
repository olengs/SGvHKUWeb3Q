from polymarket import PolyMarket, fetch_polymarket_bands
import os
import json
import dotenv

dotenv.load_dotenv()

MODE = os.getenv("MODE")
API_KEY = os.getenv(f"{MODE}_API_KEY")
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET")

if __name__ == "__main__":
    all_results, bands = fetch_polymarket_bands()

    with open("polymarket_bands.json", "w", encoding="utf-8") as f:
        json.dump(bands, f, indent=2)

    print("\n[✓] polymarket_bands.json written")