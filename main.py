import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

JOKE_API_URL = "https://official-joke-api.appspot.com/random_joke"


class HealthResponse(BaseModel):
    status: str


class JokeResponse(BaseModel):
    setup: str
    punchline: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/joke", response_model=JokeResponse)
async def joke() -> JokeResponse:
    async with httpx.AsyncClient() as client:
        response = await client.get(JOKE_API_URL)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Joke API unavailable")
    data = response.json()
    return JokeResponse(setup=data["setup"], punchline=data["punchline"])
