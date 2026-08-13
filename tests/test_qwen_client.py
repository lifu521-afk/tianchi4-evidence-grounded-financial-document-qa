from __future__ import annotations

import http.client
import io
import json
import unittest
from unittest.mock import patch

from agent.config import LLMConfig
from agent.qwen_client import OpenAICompatibleClient


class _Response:
    def __init__(self, payload: dict) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body.read()


class QwenClientRetryTests(unittest.TestCase):
    def make_client(self, *, max_retries: int = 3) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            LLMConfig(
                provider="qwen",
                api_key="test-key",
                model="test-model",
                base_url="https://example.invalid/v1",
                max_retries=max_retries,
            )
        )

    @patch("agent.qwen_client.time.sleep")
    @patch("agent.qwen_client.urllib.request.urlopen")
    def test_retries_remote_disconnect(self, mock_urlopen, mock_sleep) -> None:
        mock_urlopen.side_effect = [
            http.client.RemoteDisconnected("temporary disconnect"),
            _Response(
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }
            ),
        ]

        result = self.make_client().chat([{"role": "user", "content": "hello"}])

        self.assertEqual(result.content, "ok")
        self.assertEqual(result.usage["total_tokens"], 12)
        self.assertEqual(mock_urlopen.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("agent.qwen_client.urllib.request.urlopen")
    def test_sends_json_response_format(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _Response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )

        self.make_client().chat(
            [{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"},
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
