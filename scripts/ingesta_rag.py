import sys
from pathlib import Path

# Agregar el directorio raíz al path para poder importar app.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser, TokenTextSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Importar el custom embedding desde el servicio
from app.services.embeddings import LlamaCppEmbedding
from app.core.config import get_settings

settings = get_settings()

# ============================================
# 1. Configurar el embedding con nuestro adapter personalizado
# ============================================
Settings.embed_model = LlamaCppEmbedding(
    api_base=settings.EMBEDDING_SERVER_URL,
    model_name=settings.EMBEDDING_MODEL,
    embed_batch_size=10,  # El adapter internamente hace loop 1 a 1
    timeout=120.0,
)

# ============================================
# 2. Parsers y Splitters
# ============================================
# Parser principal: Respeta la estructura de Markdown (tablas, headers, listas)
md_parser = MarkdownNodeParser()

# Splitter secundario (Red de seguridad):
# Garantiza que ningún chunk supere los 500 tokens (margen seguro para el servidor de embeddings)
# El modelo nomic-embed soporta hasta 8192, pero si tu servidor tiene --ctx-size 512 o 2048,
# esto evita el error 400 "exceed_context_size_error"
splitter = TokenTextSplitter(
    chunk_size=450,
    chunk_overlap=50,
    separator=" ",
)

# ============================================
# 3. Cargar todos los MD automáticamente
# ============================================
DOCUMENTOS_DIR = "/opt/data_unefa"
print(f"📚 Leyendo documentos desde: {DOCUMENTOS_DIR}")
documents = SimpleDirectoryReader(DOCUMENTOS_DIR).load_data()
print(f"   → {len(documents)} documentos encontrados.")

# ============================================
# 4. Parsear respetando jerarquías
# ============================================
initial_nodes = md_parser.get_nodes_from_documents(documents)
print(f"🔪 {len(initial_nodes)} nodos iniciales generados por MarkdownNodeParser.")

# ============================================
# 5. Aplicar el splitter de seguridad
# ============================================
final_nodes = []
for node in initial_nodes:
    # El splitter subdividirá automáticamente los nodos que excedan el chunk_size
    sub_nodes = splitter.get_nodes_from_documents([node])
    final_nodes.extend(sub_nodes)

print(f"📦 {len(final_nodes)} nodos finales tras aplicar límites de tamaño.")

# ============================================
# 6. Conectar con ChromaDB (Ruta del Host)
# ============================================
CHROMA_DIR = "/opt/chroma_data"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# ¡IMPORTANTE! Borrar la colección anterior para hacer una reingesta limpia
COLLECTION_NAME = settings.CHROMA_COLLECTION_NAME
try:
    chroma_client.delete_collection(COLLECTION_NAME)
    print(f"🗑️  Colección antigua '{COLLECTION_NAME}' eliminada.")
except Exception:
    pass

chroma_collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)

# ============================================
# 7. Indexar automáticamente en ChromaDB
# ============================================
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print(f"🚀 Indexando {len(final_nodes)} chunks en ChromaDB...")
index = VectorStoreIndex(final_nodes, storage_context=storage_context)

print(
    f"✅ Ingesta completa. {chroma_collection.count()} chunks en la base de conocimiento."
)
