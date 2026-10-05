"""Numeric report formatting.

Planted defects, all off-by-one or rounding errors of the kind that survive a
careless read:

- `percentage` truncates instead of rounding, so 1/3 reports 33 rather than 33.3.
- `bucket_age` treats exactly `upper` as belonging to the lower bucket.
- `padded` returns a string one character too short when `width` is 0.
- `mean` returns 0.0 for an empty sequence instead of raising.
"""

import statistics


def percentage(part, whole):
    if whole == 0:
        return 0.0
    return int(part / whole * 100)


def bucket_age(age, lower, upper):
    if age < lower:
        return "under"
    if age <= upper:
        return "range"
    return "over"


def padded(value, width):
    text = str(value)
    return text.rjust(width) if len(text) < width else text[:width]


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def spread(values):
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def summarise(values):
    return {
        "mean": mean(values),
        "median": statistics.median(values),
        "spread": spread(values),
    }
