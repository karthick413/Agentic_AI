import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI

from agentic_ai import config


# Stores token usage, cost, latency, and whether usage was estimated.
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0
    estimated: bool = False


# Stores the complete LLM response along with its usage information.
@dataclass
class LLMResult:
    text: str
    usage: Usage


# Stores one piece of a streamed response.
# Usage is populated only in the final chunk.
@dataclass
class StreamChunk:
    text: str = ""
    usage: Usage | None = None  # set only on the final chunk


# Calculates the API cost using input and output token prices.
# Prices are specified per million tokens.
def compute_cost(
    inp: int, out: int, price_in_per_m: float, price_out_per_m: float
) -> float:
    return (inp * price_in_per_m + out * price_out_per_m) / 1_000_000


# Estimates token count when the provider does not return actual usage.
# This is a rough estimate: approximately 4 characters per token.
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough fallback: ~4 chars per token


class LLMClient:
    # Initializes the LLM client with provider configuration.
    # Optional arguments allow overriding values from config.py.
    def __init__(
        self,
        *,
        api_key=None,
        base_url=None,
        model=None,
        price_in=None,
        price_out=None,
        timeout=30.0,
        max_retries=3,
    ):

        # Use the supplied model or fall back to the configured model.
        self.model = model or config.LLM_MODEL

        # Use supplied token prices or fall back to config.py.
        self.price_in = config.PRICE_IN_PER_M if price_in is None else price_in
        self.price_out = config.PRICE_OUT_PER_M if price_out is None else price_out

        # Create the asynchronous OpenAI SDK client.
        self._client = AsyncOpenAI(
            api_key=api_key or config.LLM_API_KEY,
            base_url=base_url or config.LLM_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,  # SDK retries 429/5xx with backoff
        )

    # Builds the request parameters dynamically.
    # Uses the configured provider-specific output token parameter.
    def _kwargs(self, temperature, max_tokens) -> dict:
        kwargs: dict = {config.MAX_TOKENS_PARAM: max_tokens}

        # Add temperature only when it is not None.
        if temperature is not None:
            kwargs["temperature"] = temperature

        return kwargs

    # Collects actual usage from the provider or estimates it if unavailable.
    def _usage(self, api_usage, messages, text, latency) -> Usage:

        # Use actual token counts if the provider returned usage information.
        if api_usage is not None and api_usage.prompt_tokens is not None:
            inp, out, est = (
                api_usage.prompt_tokens,
                api_usage.completion_tokens or 0,
                False,
            )

        else:
            # Estimate input tokens from the conversation messages.
            inp = sum(estimate_tokens(m["content"]) for m in messages)

            # Estimate output tokens from the generated response.
            out, est = estimate_tokens(text), True

        # Return token counts, calculated cost, latency, and estimate status.
        return Usage(
            inp,
            out,
            compute_cost(inp, out, self.price_in, self.price_out),
            latency,
            est,
        )

    # Sends a request and waits for the complete LLM response.
    async def complete(self, messages, *, temperature=0.2, max_tokens=500) -> LLMResult:

        # Start measuring response time.
        start = time.perf_counter()

        # Send the chat completion request asynchronously.
        resp = await self._client.chat.completions.create(
            model=self.model, messages=messages, **self._kwargs(temperature, max_tokens)
        )

        # Extract the generated text from the first response choice.
        text = resp.choices[0].message.content or ""

        # Return the response text along with usage and cost information.
        return LLMResult(
            text, self._usage(resp.usage, messages, text, time.perf_counter() - start)
        )

    # Sends a request and receives the response progressively as chunks.
    async def stream(
        self, messages, *, temperature=0.2, max_tokens=500
    ) -> AsyncIterator[StreamChunk]:

        # Start measuring total streaming time.
        start = time.perf_counter()

        # Stores all received text pieces for final usage estimation.
        parts: list[str] = []

        # Stores usage information when the provider sends it.
        api_usage = None

        # Create a streaming chat completion request.
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            # Ask the provider to include token usage in the final chunk.
            # Delete this line if your provider rejects this option.
            stream_options={"include_usage": True},
            **self._kwargs(temperature, max_tokens),
        )

        # Process each chunk as it arrives from the provider.
        async for chunk in stream:
            # Capture usage information if the current chunk contains it.
            if getattr(chunk, "usage", None) is not None:
                api_usage = chunk.usage

            # Check whether the chunk contains generated text.
            if chunk.choices and chunk.choices[0].delta.content:
                # Extract the current piece of generated text.
                piece = chunk.choices[0].delta.content

                # Save the piece so the complete response can be reconstructed.
                parts.append(piece)

                # Immediately send this text piece to the caller.
                yield StreamChunk(text=piece)

        # After streaming finishes, send one final chunk containing usage.
        yield StreamChunk(
            usage=self._usage(
                api_usage,
                messages,
                "".join(parts),
                time.perf_counter() - start,
            )
        )
