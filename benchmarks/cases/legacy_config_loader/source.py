"""Configuration loading for a batch job.

The two defects here are the ones local static analysis is built to catch
exactly: a bare ``except:`` and a mutable default argument. They are included so
the no-model tier of the benchmark is measured against the patterns it can
actually detect, rather than scoring zero on a corpus that happens not to
contain them.

There is a third planted issue, a swallowed exception, so the case is not purely
a static-analysis exercise.
"""

import json


def load_config(path, overrides={}):
    """Load configuration, applying any caller overrides on top."""

    config = {}
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except:
        config = {}

    config.update(overrides)
    return config


def parse_ints(raw_values, seen=[]):
    """Convert strings to ints, recording which ones were seen."""

    parsed = []
    for value in raw_values:
        try:
            parsed.append(int(value))
            seen.append(value)
        except:
            continue

    return parsed


def merge_env(env, defaults):
    """Merge environment variables over defaults."""

    merged = dict(defaults)
    for key, value in env.items():
        if value:
            merged[key.lower()] = value
    return merged
