import json

import pytest
from fastapi.testclient import TestClient

from agentic_ai.main import TICKETS, app, get_summarizer
from agentic_ai.summarizer import StubSummarizer

TEXT = "FastAPI makes it easy to build APIs with automatic validation and docs"
TICKET = dict(
    id=1,
    title="Login fails",
    description="Cannot log in since today",
    customer_email="a@example.com",
)


@pytest.fixture
def client():
    app.dependency_overrides[get_summarizer] = lambda: StubSummarizer(delay=0)
    TICKETS.clear()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_summarize_ok(client):
    r = client.post("/summarize", json={"text": TEXT, "max_words": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["word_count"] == 5
    assert body["model"] == "stub-v0"


def test_summarize_rejects_short_text(client):
    r = client.post("/summarize", json={"text": "too short"})
    assert r.status_code == 422


def test_summarize_rejects_bad_max_words(client):
    r = client.post("/summarize", json={"text": TEXT, "max_words": 1000})
    assert r.status_code == 422


def test_stream_emits_tokens_then_done(client):
    with client.stream(
        "POST", "/summarize/stream", json={"text": TEXT, "max_words": 5}
    ) as r:
        print("STATUS:", r.status_code)
        print("CONTENT TYPE:", r.headers.get("content-type"))
        print("RESPONSE:", r.read().decode())

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")


def test_ticket_create_and_get(client):
    assert client.post("/tickets", json=TICKET).status_code == 201
    r = client.get("/tickets/1")
    assert r.status_code == 200
    assert r.json()["title"] == "Login fails"


def test_duplicate_ticket_conflict(client):
    client.post("/tickets", json=TICKET)
    assert client.post("/tickets", json=TICKET).status_code == 409


def test_missing_ticket_404(client):
    assert client.get("/tickets/999").status_code == 404


def test_stream_done_reports_word_count(client):
    with client.stream(
        "POST", "/summarize/stream", json={"text": TEXT, "max_words": 5}
    ) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    data_lines = [ln for ln in body.splitlines() if ln.startswith("data:")]
    done = json.loads(data_lines[-1][len("data:"):])
    assert done["word_count"] == 5
