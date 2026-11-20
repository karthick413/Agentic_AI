import asyncio

from agentic_ai.llm import LLMClient


async def main() -> None:
    llm = LLMClient()
    history = [{"role": "system", "content": "Be concise."}]
    for q in [
        "What is RAG?",
        "Why use it instead of fine-tuning?",
        "Give one downside.",
    ]:
        history.append({"role": "user", "content": q})
        print(f"\nQ: {q}\nA: ", end="", flush=True)
        reply = ""
        async for chunk in llm.stream(history, max_tokens=2000):
            if chunk.text:
                reply += chunk.text
                print(chunk.text, end="", flush=True)
            if chunk.usage:
                u = chunk.usage
                print(
                    f"\n[{u.input_tokens} in / {u.output_tokens} out | ${u.cost:.5f} | {u.latency:.1f}s]"
                )
        history.append({"role": "assistant", "content": reply})


asyncio.run(main())
