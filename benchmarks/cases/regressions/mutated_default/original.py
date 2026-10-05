def load_settings(path, default_port=8080):
    """Read settings, falling back to defaults for anything missing."""

    with open(path, encoding="utf-8") as handle:
        settings = {"port": default_port, "host": "localhost"}
        for line in handle:
            key, _, value = line.partition("=")
            settings[key.strip()] = value.strip()
    return settings
