"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter.

    Args:
        temperature: Sampling temperature. Lower values (e.g. 0.3) make the
            output more deterministic and stable; higher values make it more
            creative. Defaults to 0.3 (Exercise 1.2).
    """
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
    )