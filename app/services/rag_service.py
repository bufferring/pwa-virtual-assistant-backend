import asyncio
import chromadb
import logging
from typing import Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.services.embeddings import LlamaCppEmbedding
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.settings = get_settings()

        # 1. Usar nuestro adapter personalizado
        self._embedding_model = LlamaCppEmbedding(
            api_base=self.settings.EMBEDDING_SERVER_URL,
            model_name=self.settings.EMBEDDING_MODEL,
            embed_batch_size=2,
            timeout=120.0,
        )
        Settings.embed_model = self._embedding_model

        # 2. Conectar con ChromaDB persistente
        self.chroma_client = chromadb.PersistentClient(
            path=self.settings.CHROMA_PERSIST_DIR
        )
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # 3. Envolver ChromaDB en el VectorStore de LlamaIndex
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # 4. Cargar el índice desde el vector store existente
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        logger.info(
            f"RAG Service (LlamaIndex + CustomEmbedding) inicializado. "
            f"Documentos: {self.chroma_collection.count()}"
        )

    async def close(self):
        await self._embedding_model.aclose()

    async def search_context(
        self, query: str, top_k: Optional[int] = None
    ) -> list[dict]:
        """Busca los chunks más relevantes usando LlamaIndex Retriever."""
        if self.chroma_collection.count() == 0:
            logger.warning("La colección RAG está vacía")
            return []

        k = top_k or self.settings.RAG_TOP_K
        retriever = self.index.as_retriever(similarity_top_k=k)

        nodes = await asyncio.to_thread(retriever.retrieve, query)

        context_chunks = []
        for node_with_score in nodes:
            node = node_with_score.node
            distancia = 1.0 - node_with_score.score if node_with_score.score else 0.0
            context_chunks.append(
                {
                    "documento": node.get_content(),
                    "metadata": node.metadata,
                    "distancia": distancia,
                }
            )

        logger.info(f"Búsqueda RAG: {len(context_chunks)} chunks encontrados.")
        return context_chunks


# Instancia singleton
rag_service = RAGService()
