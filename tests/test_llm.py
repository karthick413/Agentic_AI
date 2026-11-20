import pytest

from agentic_ai.llm import LLMResult, StreamChunk, Usage, compute_cost
from agentic_ai.prompt_loader import load_prompt
from agentic_ai.summarizer import LLMSummarizer


class FakeLLM:
    model = "fake-model"

    def __init__(self):
        self.seen = None

    async def complete(self, messages, **kwargs):
        self.seen = messages
        return LLMResult(text="  short summary  ", usage=Usage())

    async def stream(self, messages, **kwargs):
        for piece in ("short ", "summary"):
            yield StreamChunk(text=piece)
        yield StreamChunk(usage=Usage())


def test_compute_cost():
    assert compute_cost(1_000_000, 500_000, 0.5, 2.0) == 1.5


def test_prompt_formats_word_limit():
    assert "25" in load_prompt("summarize_v2").format(max_words=25)


def test_missing_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


async def test_summarizer_builds_messages_and_strips():
    llm = FakeLLM()
    out = await LLMSummarizer(llm).summarize("some long text here", 10)
    assert out == "short summary"
    assert "10" in llm.seen[0]["content"] and "<text>" in llm.seen[1]["content"]


async def test_summarizer_streams_text_only():
    tokens = [t async for t in LLMSummarizer(FakeLLM()).stream("text", 5)]
    assert "".join(tokens) == "short summary"
