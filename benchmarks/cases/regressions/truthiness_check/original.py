def total(items):
    """Sum the `price` of every item, skipping anything missing one."""

    total = 0
    for item in items:
        if "price" in item:
            total += item["price"]
    return total
