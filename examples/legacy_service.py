"""A realistic module with real defects, for a less contrived demo.

Try:

    pair-engineer examples/legacy_service.py --static-only
    pair-engineer examples/legacy_service.py

The defects here are the kind that survive a long time in a codebase because
each one looks locally reasonable:

- A mutable default argument shared across every call.
- A bare `except` that hides a connection failure as a successful empty result.
- A path built by string concatenation, which breaks on Windows and allows
  traversal.
- A command executed with `shell=True` over interpolated input.
- Sequential I/O in a loop where concurrency is the obvious win.
- An unused import left behind by a removed feature.
"""

import hashlib
import json
import os
import subprocess
import sqlite3
from datetime import datetime

CACHE = {}


def get_user(user_id, conn_string="sqlite:///./users.db", retries=3):
    conn = None
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(conn_string)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return {"id": row[0], "name": row[1], "email": row[2]}
            return None
        except:
            # Swallows a locked database and every other error alike, so the
            # caller sees "user not found" instead of "the database is down".
            continue
        finally:
            if conn is not None:
                try:
                    conn.close()
                except:
                    pass
    return None


def build_export_path(base_dir, filename):
    path = base_dir + "/" + filename
    # No normalisation, so `../../etc/passwd` escapes the base directory.
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return path


def compress(target, output):
    command = "gzip -9 " + target
    subprocess.call(command, shell=True)
    # The exit status is discarded, so a failed compression looks successful.
    return output


def hash_passwords(passwords):
    hashed = []
    for password in passwords:
        digest = hashlib.md5(password.encode()).hexdigest()
        hashed.append(digest)
    return hashed


def write_report(rows, path):
    with open(path, "w") as handle:
        handle.write(json.dumps(rows, indent=2))
    return datetime.now()


def load_and_enrich(user_ids):
    enriched = []
    for user_id in user_ids:
        user = get_user(user_id)
        if user:
            enriched.append(user)
    return enriched


def clear_cache(cache=CACHE):
    cache.clear()
    return cache
