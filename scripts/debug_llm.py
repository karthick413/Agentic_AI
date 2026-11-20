import asyncio

from openai import APIStatusError, AsyncOpenAI

from agentic_ai import config

client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
MSG = [{"role": "user", "content": "Say hi in 3 words."}]


async def attempt(label: str, **kwargs) -> None:
    try:
        resp = await client.chat.completions.create(
            model=config.LLM_MODEL, messages=MSG, **kwargs
        )
        if kwargs.get("stream"):
            async for _ in resp:
                pass
        print(f"PASS  {label}")
    except APIStatusError as exc:
        print(f"FAIL  {label} -> {exc.status_code}: {exc.response.text[:300]}")


async def main() -> None:
    print("model:", config.LLM_MODEL, "| base_url:", config.LLM_BASE_URL)
    await attempt("1 minimal")
    await attempt("2 max_tokens", max_tokens=600)
    await attempt("3 max_completion_tokens", max_completion_tokens=600)
    await attempt("4 temperature", temperature=0.2)
    await attempt("5 stream", stream=True)
    await attempt(
        "6 stream + include_usage", stream=True, stream_options={"include_usage": True}
    )


asyncio.run(main())
