from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Paths:
    root: Path
    dataset_dir: Path
    questions_dir: Path
    raw_dir: Path
    processed_dir: Path


def default_paths(root: str | Path = ".") -> Paths:
    root_path = Path(root).resolve()
    dataset_dir = root_path / "public_dataset_a" / "public_dataset_upload"
    return Paths(
        root=root_path,
        dataset_dir=dataset_dir,
        questions_dir=dataset_dir / "questions" / "group_a",
        raw_dir=dataset_dir / "raw",
        processed_dir=root_path / "processed_data",
    )


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.6-plus"
    temperature: float = 0.0
    timeout_seconds: int = 120
    max_retries: int = 3
    provider: str = "qwen"
    enable_thinking: bool | None = None
    max_output_tokens: int = 0

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


# Backward-compatible name used by older imports.
QwenConfig = LLMConfig


LOCAL_CONFIG_MODULES = ("local_config", "llm_settings")


def load_code_settings() -> dict[str, Any]:
    """Load optional local Python settings from the project root.

    CLI arguments still have highest priority. Non-empty values in local_config.py
    override environment variables so VSCode users can run without setting env vars.
    """
    for module_name in LOCAL_CONFIG_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise
        return {name: getattr(module, name) for name in dir(module) if name.isupper()}
    return {}


def first_string(settings: dict[str, Any], *names: str) -> str:
    for name in names:
        value = settings.get(name)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""


def first_int(settings: dict[str, Any], default: int, *names: str) -> int:
    value = first_string(settings, *names)
    return int(value) if value else default


def first_float(settings: dict[str, Any], default: float, *names: str) -> float:
    value = first_string(settings, *names)
    return float(value) if value else default


def first_bool(settings: dict[str, Any], default: bool | None, *names: str) -> bool | None:
    value = first_string(settings, *names)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def llm_config_from_env(
    *,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMConfig:
    code_settings = load_code_settings()

    code_provider = first_string(code_settings, "LLM_PROVIDER", "PROVIDER", "MODEL_PROVIDER")
    provider = provider or code_provider or os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or "qwen"
    provider = provider.lower()

    code_api_key = first_string(code_settings, "LLM_API_KEY", "API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY")
    env_api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or ""
    )
    code_base_url = first_string(code_settings, "LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "QWEN_API_BASE")
    env_base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("QWEN_API_BASE")
        or LLMConfig.base_url
    )
    code_model = first_string(code_settings, "LLM_MODEL", "MODEL", "OPENAI_MODEL", "QWEN_MODEL")
    env_model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("QWEN_MODEL") or LLMConfig.model

    temperature = first_float(
        code_settings,
        float(os.getenv("LLM_TEMPERATURE") or os.getenv("QWEN_TEMPERATURE") or "0"),
        "LLM_TEMPERATURE",
        "TEMPERATURE",
        "QWEN_TEMPERATURE",
    )
    timeout_seconds = first_int(
        code_settings,
        int(os.getenv("LLM_TIMEOUT_SECONDS") or os.getenv("QWEN_TIMEOUT_SECONDS") or "120"),
        "LLM_TIMEOUT_SECONDS",
        "TIMEOUT_SECONDS",
        "QWEN_TIMEOUT_SECONDS",
    )
    max_retries = first_int(
        code_settings,
        int(os.getenv("LLM_MAX_RETRIES") or os.getenv("QWEN_MAX_RETRIES") or "3"),
        "LLM_MAX_RETRIES",
        "MAX_RETRIES",
        "QWEN_MAX_RETRIES",
    )
    enable_thinking = first_bool(
        code_settings,
        None,
        "LLM_ENABLE_THINKING",
        "ENABLE_THINKING",
        "QWEN_ENABLE_THINKING",
    )
    max_output_tokens = first_int(
        code_settings,
        int(os.getenv("LLM_MAX_OUTPUT_TOKENS") or "0"),
        "LLM_MAX_OUTPUT_TOKENS",
        "MAX_OUTPUT_TOKENS",
    )

    return LLMConfig(
        api_key=api_key or code_api_key or env_api_key,
        base_url=base_url or code_base_url or env_base_url,
        model=model or code_model or env_model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        provider=provider,
        enable_thinking=enable_thinking,
        max_output_tokens=max_output_tokens,
    )


def qwen_config_from_env() -> QwenConfig:
    return llm_config_from_env(provider="qwen")
