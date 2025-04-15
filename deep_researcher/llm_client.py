import os
from typing import Union
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel, set_tracing_export_api_key, set_tracing_disabled
from dotenv import load_dotenv
from utils.os import get_env_with_prefix

load_dotenv(override=True)

OPENAI_API_KEY = get_env_with_prefix("OPENAI_API_KEY")
DEEPSEEK_API_KEY = get_env_with_prefix("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = get_env_with_prefix("OPENROUTER_API_KEY")
GEMINI_API_KEY = get_env_with_prefix("GEMINI_API_KEY")
ANTHROPIC_API_KEY = get_env_with_prefix("ANTHROPIC_API_KEY")
PERPLEXITY_API_KEY = get_env_with_prefix("PERPLEXITY_API_KEY")
HUGGINGFACE_API_KEY = get_env_with_prefix("HUGGINGFACE_API_KEY")
LOCAL_MODEL_URL = get_env_with_prefix("LOCAL_MODEL_URL")  # e.g. "http://localhost:11434/v1"

REASONING_MODEL_PROVIDER = get_env_with_prefix("REASONING_MODEL_PROVIDER", "openai")
REASONING_MODEL = get_env_with_prefix("REASONING_MODEL", "o3-mini")
MAIN_MODEL_PROVIDER = get_env_with_prefix("MAIN_MODEL_PROVIDER", "openai")
MAIN_MODEL = get_env_with_prefix("MAIN_MODEL", "gpt-4o")
FAST_MODEL_PROVIDER = get_env_with_prefix("FAST_MODEL_PROVIDER", "openai")
FAST_MODEL = get_env_with_prefix("FAST_MODEL", "gpt-4o-mini")

supported_providers = ["openai", "deepseek", "openrouter", "gemini", "anthropic", "perplexity", "huggingface", "local"]

provider_mapping = {
    "openai": {
        "model": OpenAIResponsesModel,
        "base_url": None,
        "api_key": OPENAI_API_KEY,
    },
    "deepseek": {
        "model": OpenAIChatCompletionsModel,
        "base_url": "https://api.deepseek.com/v1",
        "api_key": DEEPSEEK_API_KEY,
    },
    "openrouter": {
        "model": OpenAIChatCompletionsModel,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
    },
    "gemini": {
        "model": OpenAIChatCompletionsModel,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_API_KEY,
    },
    "anthropic": {
        "model": OpenAIChatCompletionsModel,
        "base_url": "https://api.anthropic.com/v1/",
        "api_key": ANTHROPIC_API_KEY,
    },
    "perplexity": {
        "model": OpenAIChatCompletionsModel,
        "base_url": "https://api.perplexity.ai/chat/completions",
        "api_key": PERPLEXITY_API_KEY,
    },
    "huggingface": {
        "model": OpenAIChatCompletionsModel,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": HUGGINGFACE_API_KEY,
    },
    "local": {
        "model": OpenAIChatCompletionsModel,
        "base_url": LOCAL_MODEL_URL,
        "api_key": "ollama",  # Required by OpenAI client but not used
    }
}

if REASONING_MODEL_PROVIDER not in supported_providers:
    raise ValueError(f"Invalid model provider: {REASONING_MODEL_PROVIDER}")
if MAIN_MODEL_PROVIDER not in supported_providers:
    raise ValueError(f"Invalid model provider: {MAIN_MODEL_PROVIDER}")
if FAST_MODEL_PROVIDER not in supported_providers:
    raise ValueError(f"Invalid model provider: {FAST_MODEL_PROVIDER}")

if OPENAI_API_KEY:
    set_tracing_export_api_key(OPENAI_API_KEY)
else:
    # If no OpenAI API key is provided, disable tracing
    set_tracing_disabled(True)

# ------- SET UP REASONING MODEL -------

reasoning_client = AsyncOpenAI(
    api_key=provider_mapping[REASONING_MODEL_PROVIDER]["api_key"],
    base_url=provider_mapping[REASONING_MODEL_PROVIDER]["base_url"],
)

reasoning_model = provider_mapping[REASONING_MODEL_PROVIDER]["model"](
    model=REASONING_MODEL,
    openai_client=reasoning_client
)

# ------- SET UP MAIN MODEL -------

main_client = AsyncOpenAI(
    api_key=provider_mapping[MAIN_MODEL_PROVIDER]["api_key"],
    base_url=provider_mapping[MAIN_MODEL_PROVIDER]["base_url"],
)

main_model = provider_mapping[MAIN_MODEL_PROVIDER]["model"](
    model=MAIN_MODEL,
    openai_client=main_client
)

# ------- SET UP FAST MODEL -------

fast_client = AsyncOpenAI(
    api_key=provider_mapping[FAST_MODEL_PROVIDER]["api_key"],
    base_url=provider_mapping[FAST_MODEL_PROVIDER]["base_url"],
)

fast_model = provider_mapping[FAST_MODEL_PROVIDER]["model"](
    model=FAST_MODEL,
    openai_client=fast_client
)


def get_base_url(model: Union[OpenAIChatCompletionsModel, OpenAIResponsesModel]) -> str:
    """Utility function to get the base URL for a given model"""
    return str(model._client._base_url)


def model_supports_structured_output(model: Union[OpenAIChatCompletionsModel, OpenAIResponsesModel]) -> bool:
    """Utility function to check if a model supports structured output"""
    structured_output_providers = ["openai.com", "anthropic.com"]
    return any(provider in get_base_url(model) for provider in structured_output_providers)


__all__ = ["reasoning_model", "main_model", "fast_model", "get_base_url", "model_supports_structured_output"]
