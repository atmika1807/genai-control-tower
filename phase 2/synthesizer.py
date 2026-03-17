"""
LLM synthesis
Builds a grounded prompt from re-ranked chunks and calls the LLM.
The prompt enforces strict grounding — the model is instructed to
answer only from the provided context and say "I don't know" if
the context doesn't contain the answer.

Zero-hallucination strategy:
  1. System prompt forbids speculation beyond the context
  2. Each chunk is numbered so the model can cite sources
  3. Temperature = 0.0 for deterministic outputs
  4. Separate streaming path for /v1/query/stream
"""

import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from typing import AsyncIterator

from query.config import settings

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a precise enterprise knowledge assistant.
Answer the user's question using ONLY the information in the numbered context chunks below.
Rules:
- If the context contains the answer, respond clearly and cite the relevant chunk numbers (e.g. [1], [2]).
- If the context does NOT contain enough information, respond exactly: "I don't have enough information in the provided documents to answer this question."
- Do NOT speculate, infer, or use knowledge outside the provided context.
- Be concise. Prefer bullet points for multi-part answers.
- Never fabricate citations or chunk numbers."""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("metadata", {}).get("source_uri", "unknown")
        parts.append(f"[{i}] (source: {source})\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


class LLMSynthesizer:

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def synthesize(
        self,
        query: str,
        chunks: list[dict],
    ) -> tuple[str, dict]:
        """
        Returns (answer_text, usage_dict).
        usage_dict: {prompt_tokens, completion_tokens, total_tokens}
        """
        context = _build_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {query}"
                ),
            },
        ]

        response = await self._client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

        answer = response.choices[0].message.content or ""
        usage  = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }

        log.info(
            "synthesis_done",
            model=settings.llm_model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )
        return answer, usage

    async def synthesize_stream(
        self,
        query: str,
        chunks: list[dict],
    ) -> AsyncIterator[str]:
        """
        Yields answer text token-by-token for SSE streaming.
        Usage stats are not available mid-stream; log separately if needed.
        """
        context  = _build_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]

        stream = await self._client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
