def total(items):
    """Sum the `price` of every item, skipping anything missing one."""

    total = 0
    for item in items:
        if item.get("price"):
            total += item["price"]
    return total
