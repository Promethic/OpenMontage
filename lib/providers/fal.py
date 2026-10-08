"""Durable fal queue adapter, used instead of synchronous fal.run generation."""

import time


def queued_result(model: str, payload: dict, headers: dict) -> dict:
    import requests

    headers = {**headers, "X-Fal-No-Retry": "1"}
    submitted = requests.post(f"https://queue.fal.run/{model}", headers=headers, json=payload, timeout=30)
    submitted.raise_for_status()
    order = submitted.json()
    # The service worker's transport hook persists these URLs before returning.
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        status = requests.get(order["status_url"], headers=headers, timeout=30)
        status.raise_for_status()
        value = status.json()
        if value.get("status") == "COMPLETED":
            response = requests.get(order["response_url"], headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            if result.get("error"):
                raise RuntimeError("fal queue completed with an error")
            return result
        time.sleep(2)
    raise TimeoutError("Original fal order is still running; refresh it through Studio")
