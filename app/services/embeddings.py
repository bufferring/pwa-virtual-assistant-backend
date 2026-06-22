import httpx
from typing import Any, List, Optional
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


# ============================================
# FUNCIONES AUXILIARES DE EXTRACCIÓN ROBUSTA
# ============================================
def _flatten_to_floats(obj: Any) -> List[float]:
    """Extrae recursivamente todos los números de una estructura y los devuelve como lista plana de floats."""
    result = []
    if isinstance(obj, (float, int)):
        result.append(float(obj))
    elif isinstance(obj, list):
        for item in obj:
            result.extend(_flatten_to_floats(item))
    return result


def _find_embedding_key(obj: Any) -> Any:
    """Busca recursivamente la clave 'embedding' en cualquier nivel del JSON."""
    if isinstance(obj, dict):
        if "embedding" in obj:
            return obj["embedding"]
        for v in obj.values():
            res = _find_embedding_key(v)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = _find_embedding_key(item)
            if res is not None:
                return res
    return None


def _find_float_lists(obj: Any) -> List[List[float]]:
    """Busca recursivamente listas que contengan números (Fallback si no hay clave 'embedding')."""
    lists = []
    if isinstance(obj, list):
        if obj and isinstance(obj[0], (float, int)):
            lists.append([float(x) for x in obj])
        else:
            for item in obj:
                lists.extend(_find_float_lists(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            lists.extend(_find_float_lists(v))
    return lists


# ============================================
# ADAPTER PERSONALIZADO PARA LLAMA.CPP
# ============================================
class LlamaCppEmbedding(BaseEmbedding):
    """
    Adapter para el servidor de embeddings de llama.cpp.
    """

    _api_base: str = PrivateAttr()
    _timeout: float = PrivateAttr(default=120.0)
    _async_client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)
    _sync_client: Optional[httpx.Client] = PrivateAttr(default=None)

    def __init__(
        self,
        api_base: str,
        model_name: str,
        timeout: float = 120.0,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ):
        clean_api_base = api_base.rstrip("/").removesuffix("/v1")

        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )

        self._api_base = clean_api_base
        self._timeout = timeout

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout)
        return self._async_client

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self._timeout)
        return self._sync_client

    def _extract_embedding(self, data: Any) -> List[float]:
        """
        Extrae un solo vector de embedding de la respuesta de llama.cpp.
        Estrategia a prueba de balas que ignora el anidamiento arbitrario.
        """
        # Estrategia 1: Buscar la clave "embedding" y aplanar su contenido
        embedding_data = _find_embedding_key(data)
        if embedding_data is not None:
            flat = _flatten_to_floats(embedding_data)
            if flat:
                return flat

        # Estrategia 2 (Fallback): Buscar la lista de floats más larga en el JSON
        float_lists = _find_float_lists(data)
        if float_lists:
            return max(float_lists, key=len)

        raise ValueError(
            f"No se pudo extraer el embedding de la respuesta: {str(data)[:200]}"
        )

    # --- MÉTODOS ASYNC (Usados por FastAPI en runtime) ---

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aget_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        embeddings = await self._aget_text_embeddings([text])
        return embeddings[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        client = self._get_async_client()
        # Usamos el endpoint oficial compatible con OpenAI
        url = f"{self._api_base}/v1/embeddings"
        embeddings = []

        for text in texts:
            # Formato estándar de OpenAI para un solo texto
            payload = {"input": text, "model": self.model_name}
            response = await client.post(url, json=payload)

            # 🕵️ RAYOS X: Si falla, imprime el motivo exacto y el texto culpable
            if response.status_code != 200:
                print(f"\n❌ ERROR {response.status_code} en endpoint {url}")
                print(f"Respuesta del servidor: {response.text[:500]}")
                print(f"Texto que falló (preview): {text[:200]}...")

            response.raise_for_status()
            data = response.json()
            embeddings.append(self._extract_embedding(data))

        return embeddings

    # --- MÉTODOS SYNC (Usados por LlamaIndex durante la ingesta) ---

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._get_text_embeddings([text])
        return embeddings[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        client = self._get_async_client()
        # Usamos el endpoint oficial compatible con OpenAI
        url = f"{self._api_base}/v1/embeddings"
        embeddings = []

        for text in texts:
            # Formato estándar de OpenAI para un solo texto
            payload = {"input": text, "model": self.model_name}
            response = await client.post(url, json=payload)

            # 🕵️ RAYOS X: Si falla, imprime el motivo exacto y el texto culpable
            if response.status_code != 200:
                print(f"\n❌ ERROR {response.status_code} en endpoint {url}")
                print(f"Respuesta del servidor: {response.text[:500]}")
                print(f"Texto que falló (preview): {text[:200]}...")

            response.raise_for_status()
            data = response.json()
            embeddings.append(self._extract_embedding(data))

        return embeddings

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        client = self._get_sync_client()
        url = f"{self._api_base}/v1/embeddings"
        embeddings = []

        for text in texts:
            payload = {"input": text, "model": self.model_name}
            response = client.post(url, json=payload)

            # 🕵️ RAYOS X: Si falla, imprime el motivo exacto y el texto culpable
            if response.status_code != 200:
                print(f"\n❌ ERROR {response.status_code} en endpoint {url}")
                print(f"Respuesta del servidor: {response.text[:500]}")
                print(f"Texto que falló (preview): {text[:200]}...")

            response.raise_for_status()
            data = response.json()
            embeddings.append(self._extract_embedding(data))

        return embeddings

    async def aclose(self) -> None:
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
