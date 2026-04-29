from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_joke_returns_setup_and_punchline():
    response = client.get("/joke")
    assert response.status_code == 200
    data = response.json()
    assert "setup" in data
    assert "punchline" in data
    assert isinstance(data["setup"], str) and len(data["setup"]) > 0
    assert isinstance(data["punchline"], str) and len(data["punchline"]) > 0
