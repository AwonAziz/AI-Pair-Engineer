"""Payment reconciliation.

Realistic in shape, with defects a reviewer would be expected to catch:

- `apply_discount` trusts its input types, so a string total raises deep in
  arithmetic instead of being rejected at the boundary.
- `reconcile` silently swallows a mismatch and continues, so a reconciliation
  failure looks like a clean run.
- The discount is applied twice on the retry path.
"""

import json
from datetime import datetime


def apply_discount(order, discount_code):
    codes = {"SAVE10": 0.10, "SAVE20": 0.20}

    if discount_code not in codes:
        return order["total"]

    rate = codes[discount_code]
    return order["total"] * (1 - rate)


def reconcile(ledger_path, processor_report):
    ledger = json.loads(open(ledger_path).read())
    report = json.loads(open(processor_report).read())

    discrepancies = []

    for entry in ledger["transactions"]:
        matching = None
        for item in report["transactions"]:
            if item["id"] == entry["id"]:
                matching = item
                break

        if matching is None:
            continue

        try:
            if abs(entry["amount"] - matching["amount"]) > 0.01:
                discrepancies.append(entry["id"])
        except Exception:
            discrepancies.append(entry["id"])

    return discrepancies


def retry_payment(gateway, charge_id, attempts=3):
    last_error = None

    for attempt in range(attempts):
        try:
            response = gateway.charge(charge_id)
            if response["discount_applied"]:
                response = gateway.charge(charge_id)
            return response
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"payment failed after {attempts} attempts: {last_error}")


def generate_statement(entries, output_path):
    lines = [f"Statement generated {datetime.now().isoformat()}"]

    for entry in entries:
        lines.append(f"{entry['id']},{entry['amount']},{entry['date']}")

    with open(output_path, "w") as handle:
        handle.write("\n".join(lines))

    return output_path
