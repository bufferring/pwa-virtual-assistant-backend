from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

# 1. Configurar el embedding
Settings.embed_model = OpenAIEmbedding(
    api_base="http://127.0.0.1:8081/v1",
    model_name="nomic-embed",
    api_key="fake-key",
)

# 2. Parser que respeta la estructura de Markdown
parser = MarkdownNodeParser()

# 3. Cargar todos los MD automáticamente
documents = SimpleDirectoryReader("/opt/data_unefa").load_data()

# 4. Parsear respetando jerarquías (títulos, tablas, listas)
nodes = parser.get_nodes_from_documents(documents)

# 5. Conectar con ChromaDB
chroma_client = chromadb.PersistentClient(path="/opt/chroma_data")
chroma_collection = chroma_client.get_or_create_collection("unefa_knowledge")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 6. Indexar automáticamente
index = VectorStoreIndex(nodes, storage_context=storage_context)

print(f"✅ Ingesta completa. {len(nodes)} chunks indexados.")
