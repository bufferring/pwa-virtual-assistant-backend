# 🤖 PWA Virtual Assistant Backend - UNEFA Apure

Backend en **FastAPI** que actúa como proxy inteligente con **RAG (Retrieval-Augmented Generation)** para un LLM compatible con la API de OpenAI, exponiendo endpoints estándar (`/v1/chat/completions`, `/v1/models`, etc.) para un asistente virtual especializado en la **UNEFA Núcleo Apure**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Filosofía del proyecto: modelo local, con salida a la nube

La idea original (y el diseño por defecto) de este backend es correr **modelos locales** vía `llama-server` (llama.cpp) en tu propio servidor, manteniendo todo el procesamiento de IA bajo tu control y sin depender de terceros.

Sin embargo, **el backend no está acoplado a llama.cpp**: solo necesita un endpoint HTTP compatible con la API de OpenAI (`/v1/chat/completions`). Esto significa que si tu servidor no tiene la capacidad de hardware suficiente para correr el modelo localmente (nuestro caso 😅, el servidor es pequeño), puedes apuntar `LLAMA_SERVER_URL` a **cualquier proveedor de inferencia en la nube** compatible con OpenAI (OpenRouter, Groq, Together AI, Fireworks, DeepSeek, el propio OpenAI, etc.) sin tocar ni una línea de código de la aplicación.

> En otras palabras: la arquitectura está pensada para self-hosting total, pero es lo suficientemente flexible para degradar de forma elegante a "self-hosted RAG + LLM en la nube" cuando el hardware no alcanza. El resto del sistema (RAG, embeddings, lógica del asistente) sigue corriendo en tu servidor.

## ✨ Características

- 🔌 **API compatible con OpenAI**: Endpoints estándar `/v1/chat/completions` y `/v1/models`
- 🧠 **RAG (Retrieval-Augmented Generation)**: Respuestas basadas en documentos reales de la UNEFA
- 📚 **Base de conocimiento vectorial**: ChromaDB con embeddings de `nomic-embed-text-v2-moe`
- ⚡ **Streaming SSE nativo**: Respuestas en tiempo real desde el LLM
- 🗣️ **Texto a voz (TTS)**: Endpoint `/tts` con Edge TTS (voz en español)
- 🔁 **LLM intercambiable**: modelo local (`llama-server`) o API en la nube compatible con OpenAI, solo cambiando variables de entorno
- 🐳 **Desacoplado por capas**: RAG/Embeddings en el host (systemd), API en contenedor (Podman)
- 🔒 **Seguro**: Usuario no-root en contenedor, secrets fuera de la imagen
- 📦 **Moderno**: Gestión de paquetes con `uv`, Python 3.12, tipado estricto
- 🛡️ **HTTPS automático**: Caddy como proxy inverso con Let's Encrypt

## 🧠 ¿Qué es RAG y cómo funciona aquí?

**RAG (Retrieval-Augmented Generation)** es una técnica que combina la búsqueda de información relevante en una base de conocimientos con la generación de respuestas por un modelo de lenguaje. **Esta parte siempre corre en tu servidor**, sin importar si el LLM generador está local o en la nube.

### Flujo de una consulta con RAG

```
1. El usuario pregunta: "¿Cuándo son las inscripciones del periodo 1-2026?"

2. FastAPI convierte la pregunta en un vector (embedding) usando nomic-embed-text
   (corre en TU servidor, vía llama-server en modo --embedding)

3. ChromaDB (en TU servidor) busca los fragmentos de documentos más similares
   semánticamente

4. Se construye un "Súper Prompt" inyectando el contexto relevante:
   ┌─────────────────────────────────────────────────────────────┐
   │ System: Eres un asistente de la UNEFA Apure.               │
   │         Responde basándote en este contexto:               │
   │         [Fragmento del calendario académico 2026]          │
   │ User: ¿Cuándo son las inscripciones?                       │
   └─────────────────────────────────────────────────────────────┘

5. El prompt se envía al LLM generador (local o en la nube) y este
   genera una respuesta precisa basada en TUS documentos
```

### Ventajas de usar RAG

- ✅ **Cero alucinaciones**: el LLM responde basado en documentos reales
- ✅ **Citas verificables**: cada respuesta incluye la fuente del documento
- ✅ **Conocimiento actualizable**: solo actualizas los documentos, sin reentrenar
- ✅ **Especialización**: el asistente sabe TODO sobre la UNEFA Apure
- ✅ **Tus datos no van al LLM en la nube como entrenamiento**: solo se envía el contexto necesario en cada petición; la base de conocimiento vectorial sigue siendo tuya

## 🏗️ Arquitectura

### Opción A — Modelo local (diseño original)

```
┌──────────────────────────────────────────────────────────────────┐
│                        TU SERVIDOR                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐        ┌─────────────────────┐                │
│  │    CADDY     │───────▶│ FastAPI (Podman)    │                │
│  │  :80 / :443  │ :8000  │ --network=host      │                │
│  │  (systemd)   │        │ usuario: appuser    │                │
│  └──────────────┘        │ + ChromaDB          │                │
│                          └──────┬──────┬───────┘                │
│                                 │      │                         │
│                      HTTP :8080 │      │ HTTP :8081              │
│                                 ▼      ▼                         │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ llama-server          │  │ llama-server         │            │
│  │ (systemd)             │  │ (systemd)            │            │
│  │ Puerto 8080            │  │ Puerto 8081          │            │
│  │ LLM principal          │  │ nomic-embed-text     │            │
│  │ (genera respuestas)    │  │ (genera embeddings)  │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Opción B — LLM en la nube + RAG/embeddings locales (mi configuración actual)

Cuando el servidor no tiene recursos para correr el LLM principal (CPU/RAM/GPU insuficientes), basta con sustituir el bloque de `llama-server` (LLM) por una API en la nube. El servicio de **embeddings sigue local**, ya que es liviano (~350 MB de RAM) y mantener el RAG en tu servidor evita enviar toda tu base de conocimiento a terceros.

```
┌──────────────────────────────────────────────────────────────────┐
│                        TU SERVIDOR (pequeño)                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐        ┌─────────────────────┐                │
│  │    CADDY     │───────▶│ FastAPI (Podman)    │                │
│  │  :80 / :443  │ :8000  │ --network=host      │                │
│  │  (systemd)   │        │ usuario: appuser    │                │
│  └──────────────┘        │ + ChromaDB          │                │
│                          └──────┬──────┬───────┘                │
│                                 │      │                         │
│                      HTTPS      │      │ HTTP :8081 (local)      │
│                                 ▼      ▼                         │
│              ┌──────────────────┐  ┌──────────────────────┐    │
│              │  API en la nube  │  │ llama-server         │    │
│              │  (OpenRouter,    │  │ (systemd)             │    │
│              │  Groq, etc.)     │  │ Puerto 8081           │    │
│              │  LLM principal   │  │ nomic-embed-text      │    │
│              │  (genera         │  │ (genera embeddings,   │    │
│              │   respuestas)    │  │  se queda local)       │    │
│              └──────────────────┘  └──────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

El cambio entre ambas opciones se hace **solo con variables de entorno** (`LLAMA_SERVER_URL` y `LLM_API_KEY`), descrito en la sección de [Configuración](#️-configuración).

### ¿Por qué esta arquitectura?

- **LLM desacoplado vía HTTP**: a `LLMService` no le importa si `LLAMA_SERVER_URL` apunta a `127.0.0.1:8080` o a una API en internet, mientras hable el protocolo de OpenAI
- **Embeddings y RAG siempre locales**: tu base de conocimiento (documentos de la UNEFA) nunca sale de tu servidor para hacer las búsquedas semánticas
- **FastAPI en Podman con `--network=host`**: aislamiento del código Python, fácil despliegue, sin overhead de red
- **ChromaDB dentro del contenedor**: base de datos vectorial persistente montada como volumen
- **Caddy como proxy inverso**: HTTPS automático, HTTP/3, y manejo de conexiones persistentes para streaming

## 📋 Requisitos previos

### En el servidor host

- **Linux** (probado en Debian/Ubuntu, Fedora)
- **Podman** >= 4.0
- **Caddy** >= 2.7
- **Python** 3.12+ (solo para desarrollo local y script de ingesta)
- **uv** >= 0.4 (gestor de paquetes)
- **llama.cpp** compilado o binario oficial descargado (necesario al menos para el servicio de embeddings)
- **GPU con drivers** (NVIDIA CUDA, AMD ROCm, o Vulkan) — opcional, solo necesario si vas a correr el LLM principal localmente

### Modelos / servicios necesarios

1. **LLM principal** para generar respuestas — dos caminos posibles:
   - **Local**: un `.gguf` (ej. Qwen2.5, Llama 3, etc.) corriendo en `llama-server`
   - **En la nube**: una cuenta y API key en un proveedor compatible con OpenAI (OpenRouter, Groq, Together AI, DeepSeek, OpenAI, etc.)
2. **nomic-embed-text-v2-moe.Q8_0.gguf** (~350MB) para generar embeddings — **siempre local**, corre cómodo incluso en servidores modestos

### Instalar dependencias del sistema

```bash
# Fedora/RHEL
sudo dnf install podman caddy

# Debian/Ubuntu
sudo apt install podman caddy

# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 📁 Estructura del proyecto

```
pwa-virtual-assistant-backend/
├── app/
│   ├── api/
│   │   ├── endpoints.py      # Rutas FastAPI (chat, models, health, rag/stats)
│   │   └── tts.py            # Endpoint de texto a voz (Edge TTS)
│   ├── core/
│   │   ├── config.py         # Configuración con pydantic-settings
│   │   └── schemas.py        # Modelos Pydantic (compatibles OpenAI)
│   ├── services/
│   │   ├── llm_service.py    # Cliente HTTP hacia el LLM (local o en la nube)
│   │   └── rag_service.py    # Servicio RAG con ChromaDB + embeddings
│   └── main.py                # App FastAPI + middlewares
├── scripts/
│   └── ingesta_rag.py         # Script para poblar la base de conocimiento
├── Containerfile               # Imagen Podman optimizada con uv
├── compose.yaml                # Definición de servicios (prod y dev)
├── pyproject.toml              # Dependencias del proyecto
├── uv.lock                     # Lockfile reproducible
├── .python-version             # 3.12
└── README.md
```

## ⚙️ Configuración

### Variables de entorno

Crea un archivo `.env` en la raíz del proyecto (ya está en `.gitignore`):

```env
# ──────────────────────────────────────────────
# LLM principal (genera las respuestas)
# ──────────────────────────────────────────────
# OPCIÓN A — Modelo local con llama-server:
LLAMA_SERVER_URL=http://127.0.0.1:8080
LLM_API_KEY=

# OPCIÓN B — API en la nube compatible con OpenAI (mi caso):
# LLAMA_SERVER_URL=https://api.tu-proveedor.com
# LLM_API_KEY=tu-api-key-del-proveedor

LLAMA_TIMEOUT=300

# ──────────────────────────────────────────────
# Embeddings — se recomienda mantener SIEMPRE local
# ──────────────────────────────────────────────
EMBEDDING_SERVER_URL=http://127.0.0.1:8081
EMBEDDING_MODEL=nomic-embed

# ──────────────────────────────────────────────
# RAG - ChromaDB
# ──────────────────────────────────────────────
CHROMA_PERSIST_DIR=/app/chroma_data
CHROMA_COLLECTION_NAME=unefa_knowledge
RAG_TOP_K=4

# API Key opcional para proteger tu propio endpoint
API_KEY=tu-super-secreto-aqui

# CORS - dominios permitidos (cambia en producción)
CORS_ORIGINS=["https://tu-dominio.com"]
```

> 💡 **Nota:** `LLM_API_KEY` se envía como header `Authorization: Bearer <key>` en cada petición al LLM. Esto es lo único que necesitas para autenticarte contra la mayoría de proveedores de inferencia en la nube compatibles con OpenAI. Revisa la documentación de tu proveedor para confirmar la URL base correcta (algunos requieren que incluyas `/v1` en `LLAMA_SERVER_URL`, otros no).

### Eligiendo un proveedor en la nube

Si tu servidor no puede correr el modelo principal localmente, cualquier proveedor que exponga un endpoint `/v1/chat/completions` compatible con OpenAI funcionará sin cambios en el código. Algunas opciones comunes:

| Proveedor | Notas |
|---|---|
| OpenRouter | Acceso a múltiples modelos open-source y propietarios con una sola API key |
| Groq | Inferencia muy rápida en modelos open-source (Llama, etc.) |
| Together AI | Buen catálogo de modelos open-source |
| DeepSeek | API propia, económica |
| OpenAI | API oficial, modelos propietarios |

## 💻 Desarrollo local

### 1. Clonar e instalar

```bash
git clone https://github.com/BufferRing/pwa-virtual-assistant-backend.git
cd pwa-virtual-assistant-backend
uv sync
```

### 2. Iniciar el LLM principal

**Opción A — Local con llama-server:**

```bash
./llama-server \
    --model /ruta/a/tu/modelo.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 4096 \
    --n-gpu-layers 99
```

**Opción B — API en la nube:** no necesitas levantar nada localmente, solo configura `LLAMA_SERVER_URL` y `LLM_API_KEY` en tu `.env` apuntando al proveedor elegido.

### 3. Iniciar llama-server (Embeddings) en otra terminal

Este servicio se recomienda mantener siempre local, sin importar la opción anterior:

```bash
./llama-server \
    --model /ruta/a/nomic-embed-text-v2-moe.Q8_0.gguf \
    --host 127.0.0.1 \
    --port 8081 \
    --embedding \
    --pooling mean \
    --ctx-size 2048
```

### 4. Poblar la base de conocimiento RAG

Coloca tus documentos Markdown en `/opt/data_unefa/` y ejecuta:

```bash
uv run python scripts/ingesta_rag.py
```

### 5. Ejecutar FastAPI

```bash
uv run uvicorn app.main:app --reload
```

La API estará en `http://127.0.0.1:8000` y la documentación interactiva (OpenAPI) en `http://127.0.0.1:8000/v1/openapi.json`.

### 6. Linting y tests

```bash
uv run ruff check .          # Linter
uv run ruff format .         # Formatter
uv run pytest                # Tests
```

## 🚀 Despliegue en producción

### Paso 1: Configurar el servicio de embeddings como systemd (siempre necesario)

Crea el archivo `/etc/systemd/system/llama-embed.service`:

```ini
[Unit]
Description=LLama.cpp Embedding Server (Nomic)
After=network.target

[Service]
Type=simple
User=llama-user
Group=llama-user
WorkingDirectory=/opt/llama-server

ExecStart=/opt/llama-server/llama-server \
    --model /opt/llama-server/models/nomic-embed-text-v2-moe.Q8_0.gguf \
    --host 127.0.0.1 \
    --port 8081 \
    --embedding \
    --pooling mean \
    --ctx-size 2048 \
    --threads 4

Restart=on-failure
RestartSec=5s
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-embed
sudo systemctl status llama-embed
```

### Paso 2: Configurar el LLM principal

**Si tu servidor tiene recursos suficientes (modelo local):**

Crea `/etc/systemd/system/llama-server.service`:

```ini
[Unit]
Description=LLama.cpp Server (LLM Principal)
After=network.target

[Service]
Type=simple
User=llama-user
Group=llama-user
WorkingDirectory=/opt/llama-server

ExecStart=/opt/llama-server/llama-server \
    --model /ruta/a/tu/modelo.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 4096 \
    --n-gpu-layers 99 \
    --threads 8 \
    --batch-size 512 \
    --ubatch-size 512 \
    --flash-attn \
    --cont-batching \
    --parallel 4

Restart=on-failure
RestartSec=5s
LimitNOFILE=65536
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo systemctl status llama-server
```

**Si tu servidor es pequeño y usas una API en la nube (mi caso):**

No necesitas ningún servicio systemd adicional. Solo asegúrate de que tu `.env` de producción tenga `LLAMA_SERVER_URL` y `LLM_API_KEY` apuntando correctamente al proveedor en la nube. El resto del despliegue (pasos 3 a 6) es exactamente igual.

### Paso 3: Preparar la base de conocimiento

```bash
# Crear carpeta para documentos
sudo mkdir -p /opt/data_unefa
sudo chown $USER:$USER /opt/data_unefa

# Copiar tus documentos Markdown
cp tus_documentos.md /opt/data_unefa/

# Crear carpeta para ChromaDB
sudo mkdir -p /opt/chroma_data
sudo chown -R 1000:1000 /opt/chroma_data

# Ejecutar ingesta
uv run python scripts/ingesta_rag.py
```

### Paso 4: Construir y ejecutar el contenedor de FastAPI

```bash
# Construir imagen
podman build -t unefa-backend:latest -f Containerfile .

# Detener contenedor anterior si existe
podman stop unefa-app 2>/dev/null || true
podman rm unefa-app 2>/dev/null || true

# Ejecutar nuevo contenedor con --network=host
podman run -d \
    --name unefa-app \
    --network=host \
    --env-file .env \
    -v /opt/chroma_data:/app/chroma_data:Z \
    --restart=always \
    unefa-backend:latest
```

Alternativamente, con `compose.yaml` ya incluido en el repo:

```bash
podman compose up -d unefa-app
```

**Flags clave explicados:**
- `--env-file .env`: inyecta las variables sin embeberlas en la imagen (incluye tu `LLM_API_KEY` si usas la nube)
- `--network=host`: el contenedor usa la red del host directamente (cero overhead de red)
- `-v /opt/chroma_data:/app/chroma_data:Z`: monta la BD vectorial de forma persistente (la `:Z` es para SELinux en Fedora/RHEL)
- `--restart=always`: reinicia si crashea

### Paso 5: Configurar Caddy

Edita `/etc/caddy/Caddyfile`:

```caddyfile
tu-dominio.com {
    reverse_proxy localhost:8000 {
        # Crítico para streaming SSE
        flush_interval -1

        # Timeouts largos para respuestas de LLM
        transport http {
            keepalive 300s
            keepalive_idle_conns 10
        }
    }

    request_body {
        max_size 10MB
    }
}
```

Recargar Caddy:

```bash
sudo systemctl reload caddy
```

## 📡 Endpoints disponibles

### `POST /v1/chat/completions`

Envía mensajes al asistente. **Con RAG activo**, el sistema buscará automáticamente en la base de conocimiento el contexto relevante antes de responder, sin importar si el LLM generador está local o en la nube.

**Request:**
```json
{
  "model": "tu-modelo",
  "messages": [
    {"role": "user", "content": "¿Cuándo son las inscripciones del periodo 1-2026?"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": true
}
```

**Response (streaming):**
```
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Según"}}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":" el calendario"}}]}

data: [DONE]
```

### `GET /v1/models`

Lista los modelos disponibles en el LLM configurado (local o en la nube).

### `GET /v1/health`

Healthcheck para Caddy. Devuelve:
```json
{"status": "healthy", "llm_server": "connected"}
```

### `GET /v1/rag/stats`

Devuelve estadísticas de la base de conocimiento RAG:
```json
{
  "total_documentos": 95,
  "coleccion": "unefa_knowledge"
}
```

### `GET /tts?text=...`

Convierte texto a voz usando Edge TTS (voz `es-MX-DaliaNeural`) y devuelve un stream de audio MP3.

### `GET /`

Endpoint raíz de verificación.

### Estrategia de Chunking

Los documentos se dividen en fragmentos (chunks) siguiendo estas reglas:

- **División por secciones (`##`)**: mantiene la coherencia temática
- **Contexto heredado**: cada chunk incluye los títulos padre para dar contexto
- **Tamaño óptimo**: 300-800 tokens por chunk
- **Overlap**: 10-15% de solapamiento entre chunks para no perder contexto

## 🐛 Troubleshooting

### El contenedor no puede conectar al LLM

1. Si usas modelo **local**, verifica que los servicios estén corriendo:
   ```bash
   sudo systemctl status llama-server
   sudo systemctl status llama-embed
   ```

2. Si usas una **API en la nube**, verifica que la key sea válida y que `LLAMA_SERVER_URL` sea exactamente la URL base que pide el proveedor (algunos esperan el `/v1` incluido, otros lo agregan internamente):
   ```bash
   podman exec -it unefa-app curl https://api.tu-proveedor.com/v1/models \
     -H "Authorization: Bearer tu-api-key"
   ```

3. Prueba la conexión de embeddings (siempre local) desde dentro del contenedor:
   ```bash
   podman exec -it unefa-app curl http://127.0.0.1:8081/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input":"hola","model":"nomic-embed"}'
   ```

### RAG no encuentra contexto relevante

1. Verifica que la base de conocimiento tenga documentos:
   ```bash
   curl http://127.0.0.1:8000/v1/rag/stats
   ```

2. Si está vacía, ejecuta la ingesta nuevamente:
   ```bash
   uv run python scripts/ingesta_rag.py
   ```

### El streaming se corta o no funciona

Asegúrate de que tu Caddyfile tenga `flush_interval -1`. Sin esto, Caddy buferiza las respuestas SSE y el cliente no recibe los chunks en tiempo real. Si usas un proveedor en la nube, confirma también que este soporte `stream: true` en `/v1/chat/completions`.

### Error de permisos en ChromaDB (SELinux)

Si usas Fedora/RHEL y ves errores de permisos, asegúrate de montar el volumen con `:Z`:
```bash
-v /opt/chroma_data:/app/chroma_data:Z
```

### Ver logs

```bash
podman logs unefa-app
journalctl -u llama-embed -f
journalctl -u llama-server -f   # solo si usas LLM local
```

## 🔄 Actualizar la base de conocimiento

Para agregar o actualizar documentos:

```bash
# 1. Coloca los nuevos/actualizados Markdowns en /opt/data_unefa/
cp nuevo_documento.md /opt/data_unefa/

# 2. Ejecuta la ingesta (recrea la colección automáticamente)
uv run python scripts/ingesta_rag.py

# 3. Reinicia el contenedor para que ChromaDB recargue
podman restart unefa-app
```

## 💡 Consumo de recursos esperado

### Modo local (LLM + embeddings en tu servidor)

| Componente | RAM aprox. | CPU |
|------------|-----------|-----|
| LLM (Qwen2.5-7B) | ~4-5 GB | Alto durante inferencia |
| Embeddings (Nomic) | ~350 MB | Bajo |
| FastAPI + ChromaDB | ~200-400 MB | Bajo |
| Caddy | ~50 MB | Bajo |

**Total recomendado**: mínimo 8 GB RAM para operación fluida.

### Modo híbrido (embeddings local + LLM en la nube)

| Componente | RAM aprox. | CPU |
|------------|-----------|-----|
| Embeddings (Nomic) | ~350 MB | Bajo |
| FastAPI + ChromaDB | ~200-400 MB | Bajo |
| Caddy | ~50 MB | Bajo |
| LLM principal | 0 (corre en la nube) | 0 |

**Total recomendado**: a partir de ~1-2 GB RAM, ideal para VPS pequeños. El costo de la inferencia se traslada al proveedor de la API en la nube (revisa su modelo de precios).

## 📄 Licencia

MIT © 2026 BufferRing. Ver [LICENSE](./LICENSE) para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir cambios importantes.

---

Hecho con ❤️ usando FastAPI, llama.cpp, ChromaDB, Podman y Caddy.

**Asistente especializado en la UNEFA Núcleo Apure** 🇻🇪
