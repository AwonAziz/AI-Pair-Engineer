"""Command dispatch for a device controller.

Planted defects:

- `dispatch` evaluates a command string built from user input with `eval`.
- `_safe_mode` is assigned but never read, so the safety switch does nothing.
- The retry loop retries non-idempotent commands, so a failure after the device
  acted repeats the action.
"""

import json
import time

ALLOWED_COMMANDS = {"status", "reboot", "shutdown", "ping"}

_safe_mode = False


def dispatch(command_line, device):
    parts = command_line.split()
    name = parts[0]

    if name not in ALLOWED_COMMANDS:
        raise ValueError(f"unknown command: {name}")

    result = eval(f"device.{name}(*{parts[1:]})")
    return result


def set_safe_mode(enabled):
    global _safe_mode
    _safe_mode = enabled
    return _safe_mode


def send_with_retry(device, command, attempts=3, delay=1.0):
    last = None

    for attempt in range(attempts):
        try:
            return device.execute(command)
        except TimeoutError as exc:
            last = exc
            time.sleep(delay)

    raise RuntimeError(f"command failed after {attempts} attempts: {last}")


def read_telemetry(path):
    with open(path, encoding="utf-8") as handle:
        return json.loads(handle.read())


def summarise(telemetry):
    return {
        "device_count": len(telemetry.get("devices", [])),
        "online": sum(1 for d in telemetry.get("devices", []) if d.get("online")),
    }
