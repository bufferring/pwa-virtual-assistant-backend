import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, ORJSONResponse
from app.core.schemas import ChatCompletionRequest, Message
from app.services.llm_service import LLMService
from app.services.rag_service import rag_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

llm_service = LLMService()

# Prompt del sistema por defecto para el asistente UNEFA Apure
SYSTEM_PROMPT_BASE = """Eres un asistente virtual especializado en la Universidad Nacional Experimental Politécnica de la Fuerza Armada Nacional Bolivariana (UNEFA), con enfoque particular en el Núcleo Apure.

Tu propósito es ayudar a estudiantes, docentes y personal administrativo respondiendo preguntas sobre:
- Reglamentos (estudios de pregrado, disciplinario, código de ética)
- Planes de estudio de las carreras ofrecidas
- Calendario académico
- Información del Núcleo Apure (ubicación, contacto, carreras)
- Historia y símbolos de la UNEFA

INSTRUCCIONES IMPORTANTES:
1. Responde SIEMPRE en español.
2. Basa tus respuestas ÚNICAMENTE en el contexto proporcionado.
3. Si la información no está en el contexto, di honestamente: "No tengo esa información en mi base de conocimiento."
4. Cuando cites información, menciona la fuente (ej: "Según el Artículo 58 del Reglamento de Pregrado...").
5. Sé claro, conciso y útil."""


def construir_prompt_con_contexto(contexto_chunks: list[dict]) -> str:
    """Construye el system prompt inyectando el contexto RAG."""
    if not contexto_chunks:
        return SYSTEM_PROMPT_BASE
    
    contexto_texto = "\n\n---\n\n".join([
        f"[Fuente: {chunk['metadata'].get('fuente', 'Desconocida')} | "
        f"Sección: {chunk['metadata'].get('seccion', 'General')}]\n"
        f"{chunk['documento']}"
        for chunk in contexto_chunks
    ])
    
    return f"""{SYSTEM_PROMPT_BASE}

=== CONTEXTO DE REFERENCIA ===
{contexto_texto}
=== FIN DEL CONTEXTO ==="""


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
                    query=user_query,
                    top_k=4
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
                request.messages[i] = Message(
                    role="system", 
                    content=system_prompt
                )
                system_exists = True
                break
        
        if not system_exists:
            request.messages.insert(
                0, 
                Message(role="system", content=system_prompt)
            )
        
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
            return ORJSONResponse(content=response)

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
        "coleccion": rag_service.collection.name
    }
