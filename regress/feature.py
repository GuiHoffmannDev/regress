"""The LLM feature under test: a customer support email classifier."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from .models import ClassifierOutput, PromptConfig


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


def usage_of(response: Any) -> Usage:
    """Some OpenAI-compatible servers omit usage; treat that as zero instead of crashing."""
    usage = getattr(response, "usage", None)
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def reasoning(effort: str | None) -> dict[str, str]:
    """Only send reasoning_effort when configured; models without reasoning reject it."""
    return {"reasoning_effort": effort} if effort else {}


def build_messages(prompt: PromptConfig, email: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": prompt.system_prompt}]
    for example in prompt.few_shot:
        messages.append({"role": "user", "content": example.email})
        messages.append({"role": "assistant", "content": example.output.model_dump_json()})
    messages.append({"role": "user", "content": email})
    return messages


async def classify_email(
    email: str, prompt: PromptConfig, client: AsyncOpenAI
) -> tuple[ClassifierOutput, Usage]:
    """Classify one email with the given prompt version.

    Raises pydantic.ValidationError when the model breaks the output contract.
    """
    response = await client.chat.completions.create(
        model=prompt.model,
        temperature=prompt.temperature,
        response_format={"type": "json_object"},
        messages=build_messages(prompt, email),
        **reasoning(prompt.reasoning_effort),
    )
    content = response.choices[0].message.content or ""
    return ClassifierOutput.model_validate_json(content), usage_of(response)
