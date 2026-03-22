from datetime import datetime

class MyLogger():
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write("")

    def log(self, data):
        with open(self.filename, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp}: {data}\n")