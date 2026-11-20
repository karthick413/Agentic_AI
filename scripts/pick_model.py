import asyncio

from openai import APIStatusError, AsyncOpenAI

from agentic_ai import config

CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]


async def main() -> None:
    client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    for model in CANDIDATES:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say hi in 3 words."}],
                max_tokens=500,
            )
            text = (resp.choices[0].message.content or "").strip()
            print(f"PASS  {model:<28} -> {text[:40]!r}")
        except APIStatusError as exc:
            print(
                f"FAIL  {model:<28} -> {exc.status_code}: {exc.response.text[:120]!r}"
            )


asyncio.run(main())
