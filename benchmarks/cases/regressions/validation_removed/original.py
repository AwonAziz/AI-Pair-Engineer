def validate_email(value):
    """Return True if `value` looks like an email address."""

    if not value:
        return False
    if len(value) < 3:
        return False
    return "@" in value and "." in value.split("@")[-1]
