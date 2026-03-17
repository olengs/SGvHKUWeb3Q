from roostoo import Roostoo
import os
import dotenv
dotenv.load_dotenv()

MODE = os.getenv("MODE")
API_KEY = os.getenv(f"{MODE}_API_KEY")   # Replace with your actual API key
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET") # Replace with your actual secret key

if __name__ == "__main__":
    
    broker = Roostoo(API_KEY, SECRET_KEY)
    print(f"API_KEY: {API_KEY}")
    print(f"SECRET_KEY: {SECRET_KEY}")
    print("\n--- Checking Server Time ---")
    print(broker.check_server_time())

    print("\n--- Getting Exchange Info ---")
    info = broker.get_exchange_info()
    if info:
        print(f"Available Pairs: {list(info.get('TradePairs', {}).keys())}")

    print("\n--- Getting Market Ticker (BTC/USD) ---")
    ticker = broker.get_ticker("BTC/USD")
    if ticker:
        print(ticker.get("Data", {}).get("BTC/USD", {}))

    print("\n--- Getting Account Balance ---")
    print(broker.get_balance())

    print("\n--- Checking Pending Orders ---")
    print(broker.get_pending_count())

    # Uncomment these to test trading actions:
    # print(place_order("BTC", "BUY", 0.01, price=95000))  # LIMIT
    print(broker.place_order("BNB/USD", "BUY", 1))      
    print(broker.place_order("BNB/USD", "SELL", 1))             # MARKET       
    print(broker.query_order(pair="BNB/USD", pending_only=False))
    # print(cancel_order(pair="BNB/USD"))