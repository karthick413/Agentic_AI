import asyncio

from openai import AsyncOpenAI

from agentic_ai import config


async def main() -> None:
    client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    models = await client.models.list()
    for m in models.data:
        print(m.id)


asyncio.run(main())
