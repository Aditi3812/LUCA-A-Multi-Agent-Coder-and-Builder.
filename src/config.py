import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda

load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"


def _parse_model_names(raw_value: str | None) -> list[str]:
    if raw_value:
        models = [model.strip() for model in raw_value.split(",") if model.strip()]
        if models:
            return models

    return [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
    ]


def _is_transient_model_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "429",
            "rate limit",
            "tokens per day",
            "too many requests",
            "temporarily unavailable",
            "quota",
        )
    )


@lru_cache(maxsize=None)
def _build_client(model_name: str) -> ChatGroq:
    return ChatGroq(model_name=model_name, temperature=0.2)


class FallbackLLM:
    def __init__(self, model_names: list[str]):
        self.model_names = model_names

    def invoke(self, messages, **kwargs):
        last_error: Exception | None = None

        for model_name in self.model_names:
            try:
                return _build_client(model_name).invoke(messages, **kwargs)
            except Exception as error:
                last_error = error
                if not _is_transient_model_error(error):
                    raise error

        if last_error is not None:
            raise last_error

        raise RuntimeError("No models are configured for fallback invocation.")


_fallback_llm = FallbackLLM(_parse_model_names(os.getenv("LUCA_GROQ_MODELS")))
llm = RunnableLambda(_fallback_llm.invoke)