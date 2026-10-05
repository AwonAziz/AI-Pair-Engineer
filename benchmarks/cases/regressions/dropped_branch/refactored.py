def build_payload(user):
    """Build the outbound payload for a user."""

    payload = {"id": user["id"], "name": user["name"]}

    if user.get("email"):
        payload["email"] = user["email"].strip().lower()

    return payload
