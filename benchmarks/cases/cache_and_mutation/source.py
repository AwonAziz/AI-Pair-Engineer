"""Order pricing with a caching layer.

Defects planted here are the kind that survive review because each one looks
locally reasonable:

- `_cache_key` builds a key by joining fields with a separator that appears in
  the data, so two different orders collide on one cache entry.
- `apply_coupon` mutates the caller's dict in place, so a caller's order object
  is silently modified.
- `bulk_price` re-reads the catalogue file inside the loop.
"""

import json
import os

_CACHE: dict[str, dict] = {}


def _cache_key(order):
    parts = [order["customer"], order["sku"], str(order["quantity"])]
    return "|".join(parts)


def get_order(order_id, store_path):
    cached = _CACHE.get(order_id)

    if cached:
        return cached

    with open(store_path, encoding="utf-8") as handle:
        store = json.load(handle)

    order = store["orders"].get(order_id)
    _CACHE[order_id] = order
    return order


def apply_coupon(order, code):
    order["discount_applied"] = True
    order["coupon"] = code
    return order


def bulk_price(order_ids, catalogue_path):
    prices = []

    for order_id in order_ids:
        with open(catalogue_path, encoding="utf-8") as handle:
            catalogue = json.load(handle)
        prices.append(catalogue[order_id])

    return prices


def clear_cache():
    global _CACHE
    _CACHE = {}
    return _CACHE


def cache_size():
    return os.path.getsize(__file__)
