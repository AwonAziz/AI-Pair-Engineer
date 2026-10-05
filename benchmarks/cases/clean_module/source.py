"""Inventory reconciliation.

A deliberately clean module.

There is nothing to find here. No injection, no swallowed exception, no
duplicate side effect, no off-by-one, no unvalidated input reaching a dangerous
sink. Every branch is tested at the boundary, every resource is closed by a
context manager, and the loops are linear.

This file exists to measure false positives, which is a different question from
recall. A reviewer that scores well on detection but reports four problems here
is not usable in a pull request, and a benchmark with no clean case cannot see
the difference.

If a defect is ever added to this file, remove the `clean = true` flag and
annotate it like any other case.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    sku: str
    quantity: int
    unit_price_cents: int

    def total_cents(self) -> int:
        return self.quantity * self.unit_price_cents


def index_by_sku(items: Iterable[Item]) -> dict[str, Item]:
    """Build a lookup keyed by SKU, keeping the last item on a duplicate."""
    return {item.sku: item for item in items}


def merge_quantities(items: Iterable[Item]) -> dict[str, int]:
    """Total the quantity per SKU.

    Raises:
        ValueError: if any quantity is negative.
    """
    totals: dict[str, int] = {}

    for item in items:
        if item.quantity < 0:
            raise ValueError(f"negative quantity for SKU {item.sku!r}")
        totals[item.sku] = totals.get(item.sku, 0) + item.quantity

    return totals


def inventory_value_cents(items: Iterable[Item]) -> int:
    """Total value in integer cents, avoiding floating-point drift."""
    return sum(item.total_cents() for item in items)


def top_skus(items: Iterable[Item], limit: int) -> list[tuple[str, int]]:
    """The ``limit`` SKUs with the greatest total value.

    Ties break on SKU so the result is deterministic across runs.
    """
    if limit < 0:
        raise ValueError("limit must not be negative")

    totals: dict[str, int] = {}
    for item in items:
        totals[item.sku] = totals.get(item.sku, 0) + item.total_cents()

    ranked = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked[:limit]


def iter_low_stock(items: Iterable[Item], threshold: int) -> Iterator[Item]:
    """Yield items at or below the given quantity threshold."""
    for item in items:
        if item.quantity <= threshold:
            yield item


def reconcile_counts(counted: dict[str, int], expected: dict[str, int]) -> dict[str, int]:
    """Return the per-SKU difference between counted and expected.

    A positive value means more was counted than expected.
    """
    skus = set(counted) | set(expected)
    return {sku: counted.get(sku, 0) - expected.get(sku, 0) for sku in sorted(skus)}


def is_balanced(counted: dict[str, int], expected: dict[str, int]) -> bool:
    """Whether a reconciliation came out even.

    Comparing against a tolerance-free zero is correct here because counts are
    whole units; no rounding is involved.
    """
    return all(delta == 0 for delta in reconcile_counts(counted, expected).values())
