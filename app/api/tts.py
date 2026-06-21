import edge_tts
from fastapi.responses import StreamingResponse
from fastapi import APIRouter

router = APIRouter()


@router.get("/tts")
async def tts(text: str):
    """
    Convierte texto a voz usando Edge TTS.
    Retorna audio MP3 en streaming.
    """
    communicate = edge_tts.Communicate(text, "es-MX-DaliaNeural")

    async def audio_stream():
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and (audio_data := chunk.get("data")):
                yield audio_data

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Content-Disposition": "inline; filename=speech.mp3",
        },
    )
