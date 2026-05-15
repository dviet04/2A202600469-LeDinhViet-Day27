"""LLM factory.

Prefer an official OpenAI key (`OPENAI_API_KEY`). For backwards compatibility
fall back to `OPENROUTER_API_KEY` (which requires `LLM_BASE_URL`).
"""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Return a `ChatOpenAI` configured for OpenAI or OpenRouter.

    - If `OPENAI_API_KEY` is set, use it directly.
    - Otherwise, if `OPENROUTER_API_KEY` is set, use the base URL from
      `LLM_BASE_URL` (defaulting to OpenRouter) so existing setups keep working.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            temperature=temperature,
        )

    # Fallback for OpenRouter compatibility
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
            temperature=temperature,
        )

    raise RuntimeError("OPENAI_API_KEY or OPENROUTER_API_KEY is not set — copy .env.example to .env and set your API key")
