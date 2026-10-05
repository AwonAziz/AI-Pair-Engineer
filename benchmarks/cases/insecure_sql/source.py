"""Injected SQL injection via string formatting.

The defect: the query is built by interpolating ``user_id`` straight into the
SQL text. Any caller who controls that value controls the query.

There is a second planted issue in the same function: the cursor is opened
inside a loop and relies on garbage collection to close it, so a large result
set exhausts file descriptors.
"""

import sqlite3


def lookup_user(conn_string, user_id):
    conn = sqlite3.connect(conn_string)
    cursor = conn.cursor()

    query = f"SELECT id, name FROM users WHERE id = '{user_id}'"
    cursor.execute(query)

    row = cursor.fetchone()
    return row


def find_all(conn_string, prefixes):
    conn = sqlite3.connect(conn_string)
    cursor = conn.cursor()
    results = []

    for prefix in prefixes:
        cursor.execute(f"SELECT * FROM users WHERE name LIKE '{prefix}%'")
        results.extend(cursor.fetchall())

    return results
