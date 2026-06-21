import httpx
from typing import AsyncGenerator
from app.core.config import get_settings
from app.core.schemas import ChatCompletionRequest


class LLMService:
    def __init__(self):
        self.settings = get_settings()
        self.llama_url = f"{self.settings.LLAMA_SERVER_URL}/v1/chat/completions"
        self.models_url = f"{self.settings.LLAMA_SERVER_URL}/v1/models"
        self.health_url = f"{self.settings.LLAMA_SERVER_URL}/health"
        self.timeout = self.settings.LLAMA_TIMEOUT

        headers = {}
        if self.settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.LLM_API_KEY}"

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            http2=True,
            headers=headers,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
        )

    async def close(self):
        await self._client.aclose()

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[bytes, None]:
        """
        Hace streaming de la respuesta del LLM
        """
        payload = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        async with self._client.stream(
            "POST", self.llama_url, json=payload
        ) as response:
            if response.status_code != 200:
                error_msg = await response.aread()
                raise Exception(
                    f"LLM server error {response.status_code}: {error_msg.decode()[:500]}"
                )

            async for chunk in response.aiter_bytes(chunk_size=65536):
                yield chunk

    async def chat_completion(self, request: ChatCompletionRequest) -> dict:
        """Respuesta completa (sin streaming)"""
        payload = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        response = await self._client.post(self.llama_url, json=payload)

        # Capturar body del error para diagnóstico
        if response.status_code != 200:
            error_body = response.text[:500]
            raise Exception(f"LLM error {response.status_code}: {error_body}")

        return response.json()

    async def list_models(self) -> dict:
        response = await self._client.get(self.models_url)
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> dict:
        try:
            response = await self._client.get(self.health_url)
            if response.status_code == 200:
                return {"status": "healthy", "llm_server": "connected"}
        except (httpx.HTTPError, OSError):
            pass
        return {"status": "unhealthy", "llm_server": "disconnected"}
