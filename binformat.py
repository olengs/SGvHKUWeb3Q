import json
import re

def parse_band_label(label):
    label = label.strip().replace("$", "")

    # Less-than band, e.g. <62k
    if label.startswith("<"):
        m = re.match(r"<\s*(\d+(?:\.\d+)?)k", label)
        if not m:
            raise ValueError(f"Invalid lower-open band label: {label}")
        upper = int(float(m.group(1)) * 1000)
        return (f"<{m.group(1)}k", None, upper)

    # Greater-than band, e.g. >80k
    if label.startswith(">"):
        m = re.match(r">\s*(\d+(?:\.\d+)?)k", label)
        if not m:
            raise ValueError(f"Invalid upper-open band label: {label}")
        lower = int(float(m.group(1)) * 1000)
        return (f">{m.group(1)}k", lower, None)

    # Closed interval, e.g. 68-70k
    m = re.match(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)k", label)
    if not m:
        raise ValueError(f"Invalid closed band label: {label}")

    low = int(float(m.group(1)) * 1000)
    high = int(float(m.group(2)) * 1000)
    return (f"{m.group(1)}-{m.group(2)}k", low, high)

def sort_key(band_tuple):
    _, low, high, _ = band_tuple

    if low is None:
        return float("-inf")
    if high is None:
        return float("inf")
    return low


def bands_by_date_format(data):

    bins = {}

    for date, entries in data.items():
        day_bands = []

        for entry in entries:
            band_label = entry["b"]
            probability = entry["p"]

            normalized_label, lower, upper = parse_band_label(band_label)
            day_bands.append((normalized_label, lower, upper, probability))

        day_bands.sort(key=sort_key)
        bins[date] = day_bands

    return bins


with open("polymarket_bands.json", "r") as f:
    my_json = json.load(f)

POLYMARKET = bands_by_date_format(my_json)

print(POLYMARKET)


  