import dotenv
dotenv.load_dotenv()

from roostoo import Roostoo
import os

MODE = os.getenv("MODE")
API_KEY = os.getenv(f"{MODE}_API_KEY")   # Replace with your actual API key
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET") # Replace with your actual secret key

ASSET = "BTC/USD"

class Detection():
    def __init__(self, ticker, difference_threshold):
        self.ticker = ticker
        self.broker = Roostoo(API_KEY, SECRET_KEY)
        self.difference_threshold = difference_threshold
        self.prev = self.get_ticker_last_price()

    def update(self):
        new_price = self.get_ticker_last_price()
        ret = True
        diff = new_price - self.prev
        if abs(diff) > self.difference_threshold:
            ret = False
        self.prev = new_price
        return ret, diff
    
    def get_ticker_last_price(self):
        values = self.broker.get_ticker(self.ticker)
        try:
            return values["Data"][self.ticker]["LastPrice"]
        except:
            return None
        

# # Sample
# def warning():
#     print("WARNING!!!")

# d = Detection(ASSET, 1)
# while True:
#     if not d.update():
#         warning()
#     print(d.prev)
#     time.sleep(5)
