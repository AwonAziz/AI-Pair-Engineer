"""Intentionally imperfect sample code.

This file is the analyzer's target, not production code. It is excluded from
linting (see `per-file-ignores` in pyproject.toml) because the defects below
are the point.

Run it against the tool:

    pair-engineer examples/sample.py
    pair-engineer examples/sample.py --static-only    # no API key needed

Defects present, by design:

- `user["age"]` raises KeyError on a record missing the key.
- Four levels of nesting for what is a single filter.
- Email validation accepts any string containing "@".
- No docstring on the function; no type hints anywhere.
- A function that both filters and formats, so neither concern is testable
  on its own.
- An empty result list gives the caller no way to tell "no adults" from
  "everyone was malformed".
"""

import os
import sys


def process_users(users):
    results = []

    for user in users:
        if user["age"] >= 18:
            if user["email"] != "":
                name = user["name"].strip().lower()
                email = user["email"].strip().lower()

                if "@" in email:
                    results.append({"name": name, "email": email, "adult": True})

    return results


def summarise(users):
    total = 0
    for user in users:
        if user["role"] == "admin":
            if user["active"]:
                if user["age"] > 30:
                    total = total + 1
    return total
