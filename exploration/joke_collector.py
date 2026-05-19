"""CLI tool: fetch N jokes concurrently, log structured events, write to CSV.

Usage:
    uv run python -m exploration.joke_collector --count 20 --output jokes.csv
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from main import (
    HTTP_TIMEOUT_SECONDS,
    JOKE_API_URL,
    MAX_ATTEMPTS,
    JokeResponse,
    UpstreamError,
)

logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO, force=True)
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()


async def _get_joke_once(client: httpx.AsyncClient) -> JokeResponse:
    """Single attempt — no FastAPI HTTPException; raises UpstreamError on any non-200."""
    response = await client.get(JOKE_API_URL)
    if response.status_code >= 500:
        raise UpstreamError(f"Upstream returned {response.status_code}")
    if response.status_code != 200:
        raise UpstreamError(f"Upstream returned {response.status_code}")
    return JokeResponse.model_validate(response.json())


async def fetch_one_joke(client: httpx.AsyncClient, job_id: int) -> dict[str, object]:
    """Fetch a single joke with retry. Returns a dict with metadata for the CSV."""
    started = time.monotonic()
    attempts_used = 0
    error: str | None = None
    joke: JokeResponse | None = None

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
            retry=retry_if_exception_type((UpstreamError, httpx.RequestError)),
            reraise=False,
        ):
            with attempt:
                attempts_used = attempt.retry_state.attempt_number
                log.info("joke_collector_attempt", job_id=job_id, attempt=attempts_used)
                joke = await _get_joke_once(client)
    except RetryError as exc:
        error = f"{type(exc.last_attempt.exception()).__name__}: {exc.last_attempt.exception()}"
        log.warning("joke_collector_exhausted", job_id=job_id, attempts=attempts_used, error=error)

    duration_ms = int((time.monotonic() - started) * 1000)

    return {
        "job_id": job_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "success": joke is not None,
        "attempts": attempts_used,
        "duration_ms": duration_ms,
        "setup": joke.setup if joke else None,
        "punchline": joke.punchline if joke else None,
        "error": error,
    }


async def collect_jokes(count: int) -> list[dict[str, object]]:
    """Fan out `count` joke fetches concurrently, return all results."""
    log.info("joke_collector_starting", count=count)
    started = time.monotonic()

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        tasks = [fetch_one_joke(client, job_id=i) for i in range(count)]
        results = await asyncio.gather(*tasks)

    total_ms = int((time.monotonic() - started) * 1000)
    successes = sum(1 for r in results if r["success"])
    log.info(
        "joke_collector_done",
        count=count,
        successes=successes,
        failures=count - successes,
        total_ms=total_ms,
    )
    return list(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch N jokes and write to CSV.")
    parser.add_argument("--count", type=int, default=10, help="Number of jokes to fetch")
    parser.add_argument("--output", type=Path, default=Path("jokes.csv"), help="Output CSV path")
    args = parser.parse_args()

    results = asyncio.run(collect_jokes(args.count))

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    log.info("joke_collector_written", path=str(args.output), rows=len(df))


if __name__ == "__main__":
    main()
