import httpx
from fastapi import APIRouter, HTTPException
from app.core.schemas import ChatCompletionRequest, Message
from app.services.llm_service import LLMService
from app.services.rag_service import rag_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

llm_service = LLMService()

# Prompt del sistema por defecto para el asistente UNEFA Apure
SYSTEM_PROMPT_BASE = """Eres el asistente virtual de la UNEFA Núcleo Apure (Venezuela).

REGLAS ESTRICTAS:
- Responde SOLO usando la información del CONTEXTO proporcionado abajo.
- Si el CONTEXTO no contiene la respuesta, responde exactamente: "No tengo esa información en mi base de conocimiento."
- Cita la fuente cuando sea posible (ej: "Según nucleo-apure...").
- Responde en español, ajustando tu respuesta a la necesidad vista en la pregunta del usuario. No reserves informacion que sea util e inherente a la pregunta del usuario
- NO inventes información. /no_think"""


def construir_prompt_con_contexto(contexto_chunks: list[dict]) -> str:
    """Construye el system prompt inyectando el contexto RAG."""
    if not contexto_chunks:
        return SYSTEM_PROMPT_BASE + "\n\nCONTEXTO: (vacío)"

    # Ordenar por relevancia (distancia menor primero)
    chunks_ordenados = sorted(contexto_chunks, key=lambda x: x["distancia"])

    contexto_texto = "\n\n".join(
        [
            f"--- DOCUMENTO {i + 1} (fuente: {chunk['metadata'].get('fuente', '?')}) ---\n"
            f"{chunk['documento']}"
            for i, chunk in enumerate(chunks_ordenados)
        ]
    )

    return f"""{SYSTEM_PROMPT_BASE}

CONTEXTO:
{contexto_texto}

FIN DEL CONTEXTO. Responde la pregunta del usuario basándote ÚNICAMENTE en lo anterior."""


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint principal para chat completions.
    1. Busca contexto relevante con RAG
    2. Inyecta el contexto en el system prompt
    3. Envía al LLM
    """
    try:
        # 1. Extraer la última pregunta del usuario
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break

        # 2. Buscar contexto con RAG
        contexto_chunks = []
        if user_query:
            try:
                contexto_chunks = await rag_service.search_context(
                    query=user_query, top_k=rag_service.settings.RAG_TOP_K
                )
            except Exception as e:
                logger.error(f"Error en búsqueda RAG: {e}")
                # Continuamos sin contexto si falla RAG

        # 3. Construir system prompt con contexto
        system_prompt = construir_prompt_con_contexto(contexto_chunks)

        # 4. Inyectar/actualizar el mensaje de sistema
        system_exists = False
        for i, msg in enumerate(request.messages):
            if msg.role == "system":
                request.messages[i] = Message(role="system", content=system_prompt)
                system_exists = True
                break

        if not system_exists:
            request.messages.insert(0, Message(role="system", content=system_prompt))

        # 5. Continuar con el flujo normal al LLM
        if request.stream:
            return StreamingResponse(
                llm_service.chat_completion_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await llm_service.chat_completion(request)
            return response

    except httpx.TimeoutException:
        logger.error("Timeout connecting to LLM server")
        raise HTTPException(status_code=504, detail="LLM server timeout")
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    try:
        return await llm_service.list_models()
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.get("/health")
async def health_check():
    return await llm_service.health_check()


@router.get("/rag/stats")
async def rag_stats():
    """Endpoint para ver el estado de la base RAG."""
    return {
        "total_documentos": rag_service.collection.count(),
        "coleccion": rag_service.collection.name,
    }
