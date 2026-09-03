import httpx
import respx
from fastapi.testclient import TestClient

from main import JOKE_API_URL, MAX_ATTEMPTS, app

client = TestClient(app)


@respx.mock
def test_joke_returns_setup_and_punchline() -> None:
    respx.get(JOKE_API_URL).mock(
        return_value=httpx.Response(
            200,
            json={"id": 1, "type": "general", "setup": "Why?", "punchline": "Because."},
        )
    )

    response = client.get("/joke")

    assert response.status_code == 200
    assert response.json() == {"setup": "Why?", "punchline": "Because."}


@respx.mock
def test_joke_retries_then_returns_502_when_upstream_always_5xx() -> None:
    route = respx.get(JOKE_API_URL).mock(return_value=httpx.Response(500))

    response = client.get("/joke")

    assert response.status_code == 502
    assert route.call_count == MAX_ATTEMPTS


@respx.mock
def test_joke_retries_then_returns_502_when_upstream_unreachable() -> None:
    route = respx.get(JOKE_API_URL).mock(side_effect=httpx.ConnectError("network down"))

    response = client.get("/joke")

    assert response.status_code == 502
    assert route.call_count == MAX_ATTEMPTS


@respx.mock
def test_joke_recovers_when_upstream_fails_then_succeeds() -> None:
    route = respx.get(JOKE_API_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(
                200,
                json={"id": 1, "type": "general", "setup": "Knock knock.", "punchline": "Who?"},
            ),
        ]
    )

    response = client.get("/joke")

    assert response.status_code == 200
    assert response.json() == {"setup": "Knock knock.", "punchline": "Who?"}
    assert route.call_count == 3
