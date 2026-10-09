import os
from functools import partial

from langfuse.openai import OpenAI  # drop-in OpenAI client: every chat completion becomes a Langfuse generation
from pydantic import BaseModel

from app import config
from app.messages import RunError

MODEL_SUGGESTIONS = (
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "google/gemini-3.5-flash",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.8-flash",
)
REPAIR_PROMPT = (
    "Your previous reply was not valid JSON for the required schema:\n{error}\n"
    "Reply again with ONLY the corrected JSON object."
)


class ConfigError(Exception):
    pass


def extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("reply contains no JSON object")
    return text[start : end + 1]


class LLM:
    """Chat completions through OpenRouter; any provider is just a different model id."""

    provider = "openrouter"

    def __init__(self, model: str | None = None) -> None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ConfigError("OPENROUTER_API_KEY is not set (put it in .env)")
        self.model = model or config.LLM_MODEL
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=key,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=config.LLM_MAX_RETRIES,
        )

    def chat(self, messages: list[dict], *, name: str, metadata: dict) -> tuple[str, dict]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={"usage": {"include": True}},  # OpenRouter returns the real cost, Langfuse records it
            name=name,  # Langfuse generation name (stable, no model or run ids in it)
            metadata=metadata,
        )
        text = response.choices[0].message.content or ""
        info: dict = {"model": response.model or self.model, "reply": text[:2000]}
        if response.usage:
            info["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        return text, info

    def structured(
        self, run, agent: str, name: str, system: str, user: str, schema: type[BaseModel], attempts: int = 2
    ):
        """Ask for JSON matching `schema`; one repair round with the validation error, then a clean failure."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = "no attempt made"
        for attempt in range(1, attempts + 1):
            text = run.call(
                agent,
                "llm_call",
                name,
                partial(self.chat, messages, name=name, metadata={"agent": agent, "attempt": str(attempt)}),
                model=self.model,
                attempt=attempt,
            )
            try:
                return schema.model_validate_json(extract_json(text))
            except ValueError as error:  # pydantic.ValidationError is a ValueError too
                last_error = str(error)[:800]
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": REPAIR_PROMPT.format(error=last_error)},
                ]
        raise RunError("failed", f"{agent}: LLM returned invalid JSON after {attempts} attempts: {last_error[:300]}")
