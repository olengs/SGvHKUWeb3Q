import dotenv
dotenv.load_dotenv()

from roostoo import Roostoo
import os
import time

MODE = os.getenv("MODE")
API_KEY = os.getenv(f"{MODE}_API_KEY")   # Replace with your actual API key
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET") # Replace with your actual secret key

ASSET = "BTC/USD"

class Detection():
    def __init__(self, ticker, difference_threshold, warning_callback):
        self.ticker = ticker
        self.callback = warning_callback
        self.broker = Roostoo(API_KEY, SECRET_KEY)
        self.difference_threshold = difference_threshold
        self.prev = self.get_ticker_last_price()

    def update(self):
        new_price = self.get_ticker_last_price()
        if new_price <= self.prev - self.difference_threshold:
            self.callback()
        self.prev = new_price
        return new_price
    
    def get_ticker_last_price(self):
        values = self.broker.get_ticker(self.ticker)
        try:
            return values["Data"][self.ticker]["LastPrice"]
        except:
            return None
        

# SAMPLE
# def warning():
#     print("WARNING!!!")

# d = Detection(ASSET, 1, warning)
# while True:
#     print(d.update())
#     time.sleep(1)
