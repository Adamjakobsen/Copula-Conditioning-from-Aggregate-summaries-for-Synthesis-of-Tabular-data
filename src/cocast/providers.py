"""OpenAI-compatible HTTP adapter for vLLM and vLLM-Metal servers."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """A recorded request failure, with no automatic hidden client retries."""

    def __init__(self, message, *, retryable=True, status=None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class OpenAICompatibleProvider:
    def __init__(self, settings):
        self.settings = dict(settings)
        base = self.settings["base_url"].rstrip("/")
        if urlsplit(base).scheme not in {"http", "https"}:
            raise ValueError("base_url must use http:// or https://.")
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")

    def complete(self, messages, *, seed):
        """Return completion text and server metadata. Never truncate the prompt."""
        payload = {
            "model": self.settings["model"],
            "messages": messages,
            "temperature": self.settings["temperature"],
            "top_p": self.settings["top_p"],
            "top_k": self.settings["top_k"],
            "min_p": self.settings["min_p"],
            "max_tokens": self.settings["max_tokens"],
            "seed": seed,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.settings["response_format_json"]:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("SYNTHPSY_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(self.url, data=json.dumps(payload, allow_nan=False).encode(), headers=headers)
        try:
            with urlopen(request, timeout=self.settings.get("timeout", 600.0)) as response:
                body = response.read()
                server_header = response.headers.get("Server")
        except HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", errors="replace")
            raise ProviderError(
                f"HTTP {exc.code}: {detail}", retryable=exc.code in {408, 409, 429} or exc.code >= 500,
                status=exc.code,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Request failed: {exc}") from exc
        try:
            data = json.loads(body)
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Completion content must be a string.")
            return {
                "content": content,
                "reasoning_content": message.get("reasoning_content"),
                "usage": data.get("usage"),
                "finish_reason": choice.get("finish_reason"),
                "response_id": data.get("id"),
                "response_model": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"),
                "server_header": server_header,
            }
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise ProviderError(f"Invalid completion API response: {exc}") from exc


def get_provider(settings):
    if settings["backend"] not in {"vllm"}:
        raise ValueError("backend must be vllm.")
    return OpenAICompatibleProvider(settings)
