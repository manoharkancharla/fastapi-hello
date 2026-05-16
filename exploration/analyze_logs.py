"""Analyze the fake access log with pandas."""

from pathlib import Path

import pandas as pd

INPUT = Path(__file__).parent / "access_log.csv"
OUTPUT = Path(__file__).parent / "summary.csv"


def main() -> None:
    # 1. Read the CSV
    df = pd.read_csv(INPUT, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows")
    print(df.head())
    print(df.dtypes)
    print()

    # 2. Filter — only slow requests (>200ms)
    slow = df[df["latency_ms"] > 200]
    print(f"Slow requests: {len(slow)} of {len(df)} ({len(slow)/len(df):.1%})")
    print()

    # 3. Group by endpoint, compute aggregates
    by_endpoint = df.groupby("endpoint").agg(
        count=("latency_ms", "count"),
        mean_ms=("latency_ms", "mean"),
        p50_ms=("latency_ms", lambda s: s.quantile(0.50)),
        p95_ms=("latency_ms", lambda s: s.quantile(0.95)),
        p99_ms=("latency_ms", lambda s: s.quantile(0.99)),
        error_rate=("status_code", lambda s: (s >= 500).mean()),
    ).round(2)
    print("Per-endpoint summary:")
    print(by_endpoint)
    print()

    # 4. Group by endpoint AND status_code — two-level groupby
    by_endpoint_status = df.groupby(["endpoint", "status_code"]).size().unstack(fill_value=0)
    print("Status code distribution per endpoint:")
    print(by_endpoint_status)
    print()

    # 5. Write summary to CSV
    by_endpoint.to_csv(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()