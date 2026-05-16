"""Generate a fake HTTP access log CSV for pandas exploration."""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

ENDPOINTS = ["/health", "/joke", "/api/users", "/api/orders", "/api/checkout"]
STATUS_CODES = [200, 200, 200, 200, 200, 404, 500, 503]  # weighted toward 200
OUTPUT = Path(__file__).parent / "access_log.csv"


def generate(n_rows: int = 500) -> None:
    start = datetime(2026, 5, 12, 9, 0, 0)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "endpoint", "status_code", "latency_ms"]
        )
        writer.writeheader()
        for i in range(n_rows):
            endpoint = random.choice(ENDPOINTS)
            status = random.choice(STATUS_CODES)
            # /api/checkout is slower; errors are slower than successes
            base_latency = 200 if endpoint == "/api/checkout" else 50
            error_penalty = 100 if status >= 500 else 0
            latency = base_latency + error_penalty + random.randint(0, 300)
            writer.writerow({
                "timestamp": (start + timedelta(seconds=i * 2)).isoformat(),
                "endpoint": endpoint,
                "status_code": status,
                "latency_ms": latency,
            })


if __name__ == "__main__":
    generate()
    print(f"Wrote {OUTPUT}")