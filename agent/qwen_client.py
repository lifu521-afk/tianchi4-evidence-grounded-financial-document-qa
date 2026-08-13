from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import LLMConfig


@dataclass
class ChatResult:
    content: str
    usage: dict[str, int]
    raw: dict[str, Any]


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat/completions client.

    Works with DashScope compatible mode and most relay/base-url providers that
    expose POST /v1/chat/completions.
    """

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise RuntimeError(
                "LLM API key is missing. Set one of LLM_API_KEY, OPENAI_API_KEY, "
                "DASHSCOPE_API_KEY, or QWEN_API_KEY before running answer generation."
            )
        self.config = config

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if self.config.enable_thinking is not None:
            payload["enable_thinking"] = self.config.enable_thinking
        if self.config.max_output_tokens > 0:
            payload["max_tokens"] = self.config.max_output_tokens
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        transient_errors = (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
        )
        for attempt in range(1, self.config.max_retries + 1):
            req = urllib.request.Request(self.config.endpoint, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                    raw = json.loads(body)
                    content = raw["choices"][0]["message"]["content"]
                    usage = raw.get("usage", {}) or {}
                    return ChatResult(
                        content=content,
                        usage={
                            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                            "total_tokens": int(usage.get("total_tokens", 0) or 0),
                        },
                        raw=raw,
                    )
            except urllib.error.HTTPError as exc:
                last_error = exc
                detail = exc.read().decode("utf-8", errors="replace")
                hint = ""
                if exc.code == 401:
                    hint = (
                        " Check whether the API key matches the selected base URL. "
                        "For relay usage set LLM_BASE_URL and LLM_API_KEY together; "
                        "for DashScope set DASHSCOPE_API_KEY."
                    )
                elif exc.code == 404:
                    hint = " Check whether LLM_BASE_URL should end with /v1, not /v1/chat/completions twice."
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}{hint}")
                retryable = exc.code == 408 or exc.code == 429 or 500 <= exc.code < 600
                if not retryable:
                    break
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** attempt, 8))
            except transient_errors as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** attempt, 8))
            except (KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"Chat request failed after {self.config.max_retries} attempts "
            f"provider={self.config.provider} model={self.config.model} endpoint={self.config.endpoint}: {last_error}"
        )


# Backward-compatible name used by existing solver code.
QwenClient = OpenAICompatibleClient


def approx_token_count(text: str) -> int:
    # Rough accounting fallback when an API response does not include usage.
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other_chars = max(len(text) - chinese_chars, 0)
    return int(chinese_chars * 0.8 + other_chars / 4) + 1
