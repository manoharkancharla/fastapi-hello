import httpx
import respx
from fastapi.testclient import TestClient

from main import JOKE_API_URL, app

client = TestClient(app)


@respx.mock
def test_joke_returns_setup_and_punchline():
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
def test_joke_returns_502_when_upstream_returns_error():
    respx.get(JOKE_API_URL).mock(return_value=httpx.Response(500))

    response = client.get("/joke")

    assert response.status_code == 502
    assert "error" in response.json()["detail"].lower()


@respx.mock
def test_joke_returns_502_when_upstream_unreachable():
    respx.get(JOKE_API_URL).mock(side_effect=httpx.ConnectError("network down"))

    response = client.get("/joke")

    assert response.status_code == 502
    assert "unreachable" in response.json()["detail"].lower()
