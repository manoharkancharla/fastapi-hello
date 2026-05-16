
import httpx
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger()

app = FastAPI()

JOKE_API_URL = "https://official-joke-api.appspot.com/random_joke"
HTTP_TIMEOUT_SECONDS = 5.0
MAX_ATTEMPTS = 3


class HealthResponse(BaseModel):
    status: str


class JokeResponse(BaseModel):
    setup: str
    punchline: str


class UpstreamError(Exception):
    """Raised when the upstream joke API returns a retryable error."""


async def _get_joke_once(client: httpx.AsyncClient) -> JokeResponse:
    """Single attempt — raises UpstreamError on 5xx, returns JokeResponse on 2xx."""
    response = await client.get(JOKE_API_URL)
    if response.status_code >= 500:
        raise UpstreamError(f"Upstream returned {response.status_code}")
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Upstream returned {response.status_code}")
    return JokeResponse.model_validate(response.json())


async def fetch_joke() -> JokeResponse:
    """Fetch a joke with exponential backoff retry on transient failures."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(MAX_ATTEMPTS),
                wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
                retry=retry_if_exception_type((UpstreamError, httpx.RequestError)),
                reraise=False,
            ):
                with attempt:
                    attempt_num = attempt.retry_state.attempt_number
                    log.info(
                        "joke_fetch_attempt",
                        attempt=attempt_num,
                        max_attempts=MAX_ATTEMPTS,
                        url=JOKE_API_URL,
                    )
                    try:
                        result = await _get_joke_once(client)
                    except (UpstreamError, httpx.RequestError) as exc:
                        log.warning(
                            "joke_fetch_attempt_failed",
                            attempt=attempt_num,
                            error=str(exc),
                            error_type=type(exc).__name__,
                        )
                        raise
                    log.info(
                        "joke_fetch_succeeded",
                        attempt=attempt_num,
                        total_attempts=attempt_num,
                    )
                    return result
        except RetryError as exc:
            log.error(
                "joke_fetch_exhausted",
                max_attempts=MAX_ATTEMPTS,
                final_error=str(exc.last_attempt.exception()),
                final_error_type=type(exc.last_attempt.exception()).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Joke API failed after {MAX_ATTEMPTS} attempts: {exc.last_attempt.exception()}",
            ) from exc

    raise HTTPException(status_code=502, detail="Unreachable code in fetch_joke")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/joke", response_model=JokeResponse)
async def joke() -> JokeResponse:
    return await fetch_joke()
