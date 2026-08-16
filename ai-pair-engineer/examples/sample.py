def process_users(users):
    """Process a list of user dictionaries, returning adult users with valid email."""

    results = []

    for user in users:
        if user["age"] >= 18:
            if user["email"] != "":

                name = user["name"].strip().lower()
                email = user["email"].strip().lower()

                if "@" in email:
                    results.append({
                        "name": name,
                        "email": email,
                        "adult": True
                    })

    return results