from typing import Any, cast

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

JOKE_API_URL = "https://official-joke-api.appspot.com/random_joke"
HTTP_TIMEOUT_SECONDS = 5.0


class HealthResponse(BaseModel):
    status: str


class JokeResponse(BaseModel):
    setup: str
    punchline: str


async def fetch_joke() -> dict[str, Any]:
    """Fetch a joke from the upstream API. Raises HTTPException on failure."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        try:
            response = await client.get(JOKE_API_URL)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Joke API unreachable: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Joke API returned an error")

    return cast(dict[str, Any], response.json())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/joke", response_model=JokeResponse)
async def joke() -> JokeResponse:
    data = await fetch_joke()
    return JokeResponse(setup=data["setup"], punchline=data["punchline"])
