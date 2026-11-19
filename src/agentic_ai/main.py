import json
from collections.abc import AsyncIterator
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from agentic_ai.models import Ticket
from agentic_ai.schemas import SummarizeRequest, SummarizeResponse
from agentic_ai.summarizer import StubSummarizer

app = FastAPI(title="Agentic AI Service", version="0.1.0")

# In-memory store, only for practice. A real service would use a database.
TICKETS: dict[int, Ticket] = {}


@lru_cache
def get_summarizer() -> StubSummarizer:
    return StubSummarizer()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    req: SummarizeRequest,
    summarizer: StubSummarizer = Depends(get_summarizer),
) -> SummarizeResponse:
    summary = await summarizer.summarize(req.text, req.max_words)
    return SummarizeResponse(
        summary=summary,
        word_count=len(summary.split()),
        model=summarizer.model_name,
    )


def sse(data: dict, event: str | None = None) -> str:
    """Format one Server-Sent Event."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"


@app.post("/summarize/stream")
async def summarize_stream(
    req: SummarizeRequest,
    summarizer: StubSummarizer = Depends(get_summarizer),
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        count = 0
        async for token in summarizer.stream(req.text, req.max_words):
            count += 1
            yield sse({"token": token})
        yield sse({"word_count": count, "model": summarizer.model_name}, event="done")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/tickets", response_model=Ticket, status_code=201)
async def create_ticket(ticket: Ticket) -> Ticket:
    if ticket.id in TICKETS:
        raise HTTPException(
            status_code=409, detail=f"Ticket {ticket.id} already exists"
        )
    TICKETS[ticket.id] = ticket
    return ticket


@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: int) -> Ticket:
    if ticket_id not in TICKETS:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TICKETS[ticket_id]
