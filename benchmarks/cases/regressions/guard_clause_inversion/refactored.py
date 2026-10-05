def get_active_users(conn, min_age):
    """Return the emails of active users at or above `min_age`."""

    cursor = conn.cursor()
    cursor.execute("SELECT email, age, active FROM users")

    results = []
    for email, age, active in cursor:
        if not active:
            continue
        if not age >= min_age:
            continue
        results.append(email)

    return results
