"""
Script de ingesta para alimentar la base de conocimiento RAG.
Ejecutar desde el servidor host (no desde el contenedor).
"""
import re
import asyncio
import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path

# Configuración
EMBEDDING_URL = "http://127.0.0.1:8081/v1/embeddings"
EMBEDDING_MODEL = "nomic-embed"
CHROMA_DIR = "/opt/chroma_data"
DOCUMENTOS_DIR = "/opt/data_unefa"
COLLECTION_NAME = "unefa_knowledge"

# Tamaño de chunk y overlap
MAX_CHUNK_SIZE = 1500  # caracteres
OVERLAP_SIZE = 200


async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            EMBEDDING_URL,
            json={"input": text, "model": EMBEDDING_MODEL}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


def chunk_por_secciones(markdown: str, fuente: str) -> list[dict]:
    """
    Divide el markdown en chunks preservando la jerarquía de títulos.
    Estrategia: dividir por ## manteniendo contexto del # padre.
    """
    chunks = []
    lineas = markdown.split('\n')
    
    titulo_padre = ""
    seccion_actual = []
    seccion_titulo = ""
    
    for linea in lineas:
        # Detectar título de nivel 1 (#)
        if linea.startswith('# ') and not linea.startswith('## '):
            titulo_padre = linea[2:].strip()
            continue
        
        # Detectar título de nivel 2 (##) - punto de corte
        if linea.startswith('## ') and not linea.startswith('### '):
            # Guardar sección anterior si existe
            if seccion_actual:
                texto = '\n'.join(seccion_actual).strip()
                if len(texto) > 50:  # Solo chunks con contenido real
                    chunks.append({
                        "texto": texto,
                        "metadata": {
                            "fuente": fuente,
                            "seccion": seccion_titulo or titulo_padre,
                            "tipo": "seccion"
                        }
                    })
            
            seccion_titulo = linea[3:].strip()
            seccion_actual = [linea]
            continue
        
        seccion_actual.append(linea)
    
    # Guardar última sección
    if seccion_actual:
        texto = '\n'.join(seccion_actual).strip()
        if len(texto) > 50:
            chunks.append({
                "texto": texto,
                "metadata": {
                    "fuente": fuente,
                    "seccion": seccion_titulo or titulo_padre,
                    "tipo": "seccion"
                }
            })
    
    # Si hay chunks muy grandes, subdividirlos
    chunks_finales = []
    for chunk in chunks:
        if len(chunk["texto"]) > MAX_CHUNK_SIZE:
            sub_chunks = subdividir_chunk(chunk)
            chunks_finales.extend(sub_chunks)
        else:
            chunks_finales.append(chunk)
    
    return chunks_finales


def subdividir_chunk(chunk: dict) -> list[dict]:
    """Subdivide un chunk grande en partes más pequeñas con overlap."""
    texto = chunk["texto"]
    sub_chunks = []
    inicio = 0
    parte_num = 1
    
    while inicio < len(texto):
        fin = min(inicio + MAX_CHUNK_SIZE, len(texto))
        
        # Intentar cortar en un límite de párrafo
        if fin < len(texto):
            ultimo_parrafo = texto.rfind('\n\n', inicio, fin)
            if ultimo_parrafo > inicio + 500:
                fin = ultimo_parrafo
        
        parte_texto = texto[inicio:fin].strip()
        if parte_texto:
            sub_chunk = {
                "texto": parte_texto,
                "metadata": {
                    **chunk["metadata"],
                    "parte": parte_num
                }
            }
            sub_chunks.append(sub_chunk)
            parte_num += 1
        
        inicio = fin - OVERLAP_SIZE if fin < len(texto) else len(texto)
    
    return sub_chunks


async def ingestar_archivo(
    archivo: Path, 
    collection, 
    client: httpx.AsyncClient
):
    """Procesa un archivo markdown y lo ingesta en ChromaDB."""
    print(f"\n📄 Procesando: {archivo.name}")
    
    contenido = archivo.read_text(encoding='utf-8')
    fuente = archivo.stem  # nombre sin extensión
    
    # Estrategia de chunking
    chunks = chunk_por_secciones(contenido, fuente)
    print(f"   → Generados {len(chunks)} chunks")
    
    # Ingestar cada chunk
    for i, chunk in enumerate(chunks):
        doc_id = f"{fuente}_chunk_{i}"
        
        try:
            # Obtener embedding
            response = await client.post(
                EMBEDDING_URL,
                json={"input": chunk["texto"], "model": EMBEDDING_MODEL}
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            
            # Guardar en ChromaDB
            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk["texto"]],
                metadatas=[chunk["metadata"]]
            )
            
            if (i + 1) % 5 == 0:
                print(f"   ✓ {i + 1}/{len(chunks)} chunks ingestados")
        
        except Exception as e:
            print(f"   ❌ Error en chunk {i}: {e}")


async def main():
    print("🚀 Iniciando ingesta RAG para UNEFA Apure...")
    
    # Configurar ChromaDB
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    # Eliminar colección anterior (para empezar limpio)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Colección '{COLLECTION_NAME}' eliminada")
    except Exception:
        pass
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Procesar todos los archivos
    documentos_dir = Path(DOCUMENTOS_DIR)
    archivos = list(documentos_dir.glob('*.md'))
    
    print(f"\n📚 Archivos encontrados: {len(archivos)}")
    for arch in archivos:
        print(f"   - {arch.name}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for archivo in archivos:
            await ingestar_archivo(archivo, collection, client)
    
    print(f"\n✅ Ingesta completada!")
    print(f"📊 Total de chunks en colección: {collection.count()}")


if __name__ == "__main__":
    asyncio.run(main())
