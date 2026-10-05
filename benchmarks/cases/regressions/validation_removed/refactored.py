def validate_email(value):
    """Return True if `value` looks like an email address."""

    if not value:
        return False
    return "@" in value
