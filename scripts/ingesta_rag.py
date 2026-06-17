__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import asyncio
import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
import re

# Configuración
EMBEDDING_URL = "http://127.0.0.1:8081/v1/embeddings"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe.Q8_0.gguf"
CHROMA_DIR = "/opt/chroma_data"
DOCUMENTOS_DIR = "/opt/data_unefa"
COLLECTION_NAME = "unefa_knowledge"

MAX_CHUNK_SIZE = 1200  # Tamaño objetivo por chunk (caracteres)
MAX_EMBED_SIZE = 2500 # Límite seguro para el servidor de embeddings
OVERLAP_SIZE = 150  # Solapamiento al subdividir chunks grandes
MAX_RETRIES = 3  # Reintentos por chunk
RETRY_DELAY = 2  # Segundos entre reintentos
REQUEST_DELAY = 0.3  # Segundos entre requests (reducido para acelerar ingesta)
MIN_CHUNK_LENGTH = 50  # Chunks más cortos que esto se omiten


async def get_embedding(
    text: str, client: httpx.AsyncClient, retries: int = MAX_RETRIES
) -> list[float]:
    """
    Obtiene embedding con reintentos automáticos.
    Muestra el cuerpo del error en caso de 400 para diagnóstico.
    """
    for attempt in range(retries):
        try:
            response = await client.post(
                EMBEDDING_URL,
                json={"input": text, "model": EMBEDDING_MODEL},
                timeout=30.0,
            )

            if response.status_code != 200:
                error_body = response.text[:500]
                if attempt < retries - 1:
                    print(
                        f"      ⚠️  Error {response.status_code}, reintentando... ({error_body[:120]})"
                    )
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                else:
                    raise Exception(f"HTTP {response.status_code}: {error_body}")

            data = response.json()
            return data["data"][0]["embedding"]

        except Exception as e:
            if attempt < retries - 1 and "HTTP" not in str(e):
                print(
                    f"      ⚠️  Reintentando ({attempt + 1}/{retries})... {str(e)[:80]}"
                )
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise e


def limpiar_texto(texto: str) -> str:
    """
    Limpia el texto para que sea compatible con el servidor de embeddings.
    SE APLICA A TODOS LOS CHUNKS, no solo a los subdivididos.
    """
    if not texto:
        return ""

    # 1. Eliminar pipes de tablas markdown (confunden al tokenizer)
    texto = texto.replace("|", " ")

    # 2. Eliminar caracteres unicode no imprimibles y no ASCII básicos
    #    Manteniendo tildes y eñes (rango unicode del español)
    texto = "".join(c for c in texto if c.isprintable() or c in "\n\t")

    # 3. Eliminar secuencias de markdown ruidosas (---, ***, ___)
    texto = re.sub(r"^[-*_]{3,}\s*$", "", texto, flags=re.MULTILINE)

    # 4. Normalizar espacios múltiples, pero preservando saltos de párrafo
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    # 5. Limitar longitud máxima (seguridad para el servidor de embeddings)
    if len(texto) > MAX_EMBED_SIZE:
        # Intentar cortar en un límite de párrafo
        corte = texto.rfind("\n\n", 0, MAX_EMBED_SIZE)
        if corte > MAX_EMBED_SIZE // 2:
            texto = texto[:corte]
        else:
            # Cortar en límite de oración
            corte = texto.rfind(". ", 0, MAX_EMBED_SIZE)
            if corte > MAX_EMBED_SIZE // 2:
                texto = texto[: corte + 1]
            else:
                texto = texto[:MAX_EMBED_SIZE]

    return texto.strip()


def chunk_por_secciones(markdown: str, fuente: str) -> list[dict]:
    """
    Divide el markdown preservando la jerarquía.

    Estrategia:
    - Solo los encabezados de nivel 2 (##) generan un nuevo chunk.
    - Los encabezados de nivel 3 (###) y 4 (####) se ACUMULAN dentro del chunk
      de la sección padre, manteniendo el contexto completo.
    - Si un chunk excede MAX_CHUNK_SIZE, se subdivide con overlap.
    - Si el documento completo cabe en MAX_CHUNK_SIZE, se retorna como un solo chunk.
    """
    # EARLY RETURN: Documentos pequeños no se dividen
    # Se limpia primero para medir el tamaño real que irá al embedding
    texto_limpio = limpiar_texto(markdown)
    if len(texto_limpio) <= MAX_CHUNK_SIZE and len(texto_limpio) > MIN_CHUNK_LENGTH:
        return [{
            "texto": texto_limpio,
            "metadata": {
                "fuente": fuente,
                "seccion": fuente,
                "tipo": "documento_completo"
            }
        }]

    # Si llega aquí, el documento es grande → aplicar chunking jerárquico
    chunks = []
    lineas = markdown.split("\n")

    titulo_padre = ""  # Nivel #
    titulo_seccion = ""  # Nivel ##
    seccion_buffer = []  # Líneas acumuladas de la sección actual

    def guardar_seccion():
        nonlocal seccion_buffer
        if not seccion_buffer:
            return
        texto = "\n".join(seccion_buffer).strip()
        if len(texto) > MIN_CHUNK_LENGTH:
            chunks.append(
                {
                    "texto": texto,
                    "metadata": {
                        "fuente": fuente,
                        "seccion": titulo_seccion or titulo_padre or fuente,
                        "tipo": "seccion",
                    },
                }
            )
        seccion_buffer = []

    for linea in lineas:
        # Nivel 1 (#) - nuevo documento/capítulo
        if linea.startswith("# ") and not linea.startswith("## "):
            guardar_seccion()
            titulo_padre = linea[2:].strip()
            titulo_seccion = ""
            seccion_buffer.append(linea)
            continue

        # Nivel 2 (##) - NUEVA SECCIÓN (punto de corte principal)
        if linea.startswith("## ") and not linea.startswith("### "):
            guardar_seccion()
            titulo_seccion = linea[3:].strip()
            seccion_buffer.append(linea)
            continue

        # Nivel 3 (###), nivel 4 (####) y demás:
        # SE ACUMULAN en la sección actual (NO cortan el chunk)
        seccion_buffer.append(linea)

    # Guardar última sección
    guardar_seccion()

    # Subdividir chunks que aún sean muy grandes
    chunks_finales = []
    for chunk in chunks:
        if len(chunk["texto"]) > MAX_CHUNK_SIZE:
            sub_chunks = subdividir_chunk(chunk)
            chunks_finales.extend(sub_chunks)
        else:
            chunks_finales.append(chunk)

    return chunks_finales


def subdividir_chunk(chunk: dict) -> list[dict]:
    """
    Subdivide un chunk grande en partes más pequeñas con overlap.
    Intenta cortar en límites de párrafo para mantener coherencia.
    """
    texto = chunk["texto"]
    sub_chunks = []
    inicio = 0
    parte_num = 1

    while inicio < len(texto):
        fin = min(inicio + MAX_CHUNK_SIZE, len(texto))

        # Intentar cortar en un límite de párrafo si estamos en medio del texto
        if fin < len(texto):
            ultimo_parrafo = texto.rfind("\n\n", inicio, fin)
            if ultimo_parrafo > inicio + (MAX_CHUNK_SIZE // 2):
                fin = ultimo_parrafo

        parte_texto = texto[inicio:fin].strip()

        if parte_texto:
            sub_chunk = {
                "texto": parte_texto,
                "metadata": {**chunk["metadata"], "parte": parte_num},
            }
            sub_chunks.append(sub_chunk)
            parte_num += 1

        # Avanzar con overlap
        if fin >= len(texto):
            break
        inicio = fin - OVERLAP_SIZE

    return sub_chunks


async def ingestar_archivo(archivo: Path, collection, client: httpx.AsyncClient):
    """Procesa un archivo markdown y lo ingesta en ChromaDB."""
    print(f"\n📄 Procesando: {archivo.name}")

    contenido = archivo.read_text(encoding="utf-8")
    fuente = archivo.stem

    chunks = chunk_por_secciones(contenido, fuente)
    print(f"   → Generados {len(chunks)} chunks")

    exitosos = 0
    fallidos = 0
    omitidos = 0

    for i, chunk in enumerate(chunks):
        # LIMPIAR SIEMPRE el texto antes de enviarlo al embedding
        texto_limpio = limpiar_texto(chunk["texto"])

        # Saltar chunks demasiado cortos (ruido)
        if len(texto_limpio) < MIN_CHUNK_LENGTH:
            omitidos += 1
            continue

        doc_id = f"{fuente}_chunk_{i}"

        try:
            # Obtener embedding con reintentos
            embedding = await get_embedding(texto_limpio, client)

            # Guardar en ChromaDB (texto limpio, no original)
            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[texto_limpio],
                metadatas=[chunk["metadata"]],
            )

            exitosos += 1

            # Delay entre requests para no sobrecargar
            await asyncio.sleep(REQUEST_DELAY)

            if (i + 1) % 5 == 0:
                print(f"   ✓ {i + 1}/{len(chunks)} chunks ingestados")

        except Exception as e:
            fallidos += 1
            preview = texto_limpio[:120].replace("\n", " ")
            print(f"   ❌ Error en chunk {i}: {e}")
            print(f"      Preview: {preview}...")

    if fallidos > 0 or omitidos > 0:
        print(
            f"   ⚠️  {exitosos} exitosos, {fallidos} fallidos, {omitidos} omitidos (muy cortos)"
        )
    else:
        print(f"   ✅ {exitosos} chunks ingestados correctamente")


async def main():
    print("🚀 Iniciando ingesta RAG para UNEFA Apure...")

    # Configurar ChromaDB
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False)
    )

    # Eliminar colección anterior (reingesta completa)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Colección '{COLLECTION_NAME}' eliminada")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    # Procesar todos los archivos markdown
    documentos_dir = Path(DOCUMENTOS_DIR)
    archivos = sorted(documentos_dir.glob("*.md"))

    print(f"\n📚 Archivos encontrados: {len(archivos)}")
    for arch in archivos:
        print(f"   - {arch.name}")

    async with httpx.AsyncClient() as client:
        for archivo in archivos:
            await ingestar_archivo(archivo, collection, client)

    print(f"\n✅ Ingesta completada!")
    print(f"📊 Total de chunks en colección: {collection.count()}")


if __name__ == "__main__":
    asyncio.run(main())
