import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.settings = get_settings()
        
        # Cliente HTTP para embeddings
        self._http_client = httpx.AsyncClient(timeout=30.0)
        
        # Cliente ChromaDB persistente
        self.chroma_client = chromadb.PersistentClient(
            path=self.settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Obtener o crear la colección
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(
            f"RAG Service inicializado. Documentos en colección: {self.collection.count()}"
        )

    async def close(self):
        await self._http_client.aclose()

    async def get_embedding(self, text: str) -> list[float]:
        """Obtiene el vector de embedding de un texto usando nomic-embed."""
        payload = {
            "input": text,
            "model": self.settings.EMBEDDING_MODEL
        }
        
        response = await self._http_client.post(
            f"{self.settings.EMBEDDING_SERVER_URL}/v1/embeddings",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    async def search_context(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        where: Optional[dict] = None
    ) -> list[dict]:
        """
        Busca los chunks más relevantes para una consulta.
        Retorna lista de dicts con 'documento', 'metadata' y 'distancia'.
        """
        if self.collection.count() == 0:
            logger.warning("La colección RAG está vacía")
            return []
        
        k = top_k or self.settings.RAG_TOP_K
        query_embedding = await self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        context_chunks = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                context_chunks.append({
                    "documento": doc,
                    "metadata": meta,
                    "distancia": dist
                })
        
        logger.info(f"Búsqueda RAG: {len(context_chunks)} chunks encontrados para query")
        return context_chunks

    async def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict
    ):
        """Añade un documento a la colección (usado por el script de ingesta)."""
        embedding = await self.get_embedding(text)
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )


# Instancia singleton
rag_service = RAGService()
