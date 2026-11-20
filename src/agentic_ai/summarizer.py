import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from agentic_ai.llm import LLMClient
from agentic_ai.prompt_loader import load_prompt


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


# Protocol defines the common interface that all summarizers must follow.
# Both StubSummarizer and LLMSummarizer can follow this interface.
class Summarizer(Protocol):
    # Returns the name of the LLM model being used.
    @property
    def model_name(self) -> str: ...

    # Asynchronously generates a complete summary.
    async def summarize(self, text: str, max_words: int) -> str: ...

    # Generates the summary progressively as text chunks.
    def stream(self, text: str, max_words: int) -> AsyncIterator[str]: ...


# Real summarizer that uses an LLM instead of a predefined response.
class LLMSummarizer:
    # Receive an LLMClient and the prompt file to use.
    def __init__(self, llm: LLMClient, prompt_name: str = "summarize_v2"):
        self.llm = llm
        self.prompt_name = prompt_name

    # Return the model name from the LLM client.
    @property
    def model_name(self) -> str:
        return self.llm.model

    # Prepare the system and user messages for the LLM.
    def _messages(self, text: str, max_words: int) -> list[dict]:

        # Load the selected prompt and replace {max_words} with its value.
        system = load_prompt(self.prompt_name).format(max_words=max_words)

        # Return messages using the system and user roles.
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"<text>\n{text}\n</text>"},
        ]

    # Calculate a suitable output token limit.
    # Extra tokens leave room for thinking models.
    def _max_tokens(self, max_words: int) -> int:
        return max(300, max_words * 6)  # leave room for thinking models

    # Generate a complete summary using the LLM.
    async def summarize(self, text: str, max_words: int) -> str:

        # Send the messages to the LLM and wait for the complete response.
        result = await self.llm.complete(
            self._messages(text, max_words), max_tokens=self._max_tokens(max_words)
        )

        # Return the summary after removing surrounding whitespace.
        return result.text.strip()

    # Generate a summary progressively using streaming.
    async def stream(self, text: str, max_words: int) -> AsyncIterator[str]:

        # Request streamed output from the LLM.
        async for chunk in self.llm.stream(
            self._messages(text, max_words), max_tokens=self._max_tokens(max_words)
        ):
            # Yield only chunks containing text.
            if chunk.text:
                yield chunk.text
