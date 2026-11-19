import asyncio
from collections.abc import AsyncIterator


class StubSummarizer:
    """Pretends to be an LLM: returns the first N words. Replaced on day 4."""

    model_name = "stub-v0"

    def __init__(self, delay: float = 0.05):
        self.delay = delay

    def _words(self, text: str, max_words: int) -> list[str]:
        return text.split()[:max_words]

    async def summarize(self, text: str, max_words: int) -> str:
        await asyncio.sleep(self.delay)  # simulate network latency
        return " ".join(self._words(text, max_words))

    async def stream(self, text: str, max_words: int) -> AsyncIterator[str]:
        for word in self._words(text, max_words):
            await asyncio.sleep(self.delay)
            yield word + " "
