# 🤖 PWA Virtual Assistant Backend - UNEFA Apure

Backend en **FastAPI** que actúa como proxy inteligente con **RAG (Retrieval-Augmented Generation)** para `llama-server` (llama.cpp), exponiendo una API compatible con OpenAI para un asistente virtual especializado en la **UNEFA Núcleo Apure**.

Diseñado para **producción** con streaming SSE, RAG con ChromaDB, arquitectura desacoplada y despliegue con Podman + systemd + Caddy.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Características

- 🔌 **API compatible con OpenAI**: Endpoints estándar `/v1/chat/completions` y `/v1/models`
- 🧠 **RAG (Retrieval-Augmented Generation)**: Respuestas basadas en documentos reales de la UNEFA
- 📚 **Base de conocimiento vectorial**: ChromaDB con embeddings de `nomic-embed-text-v2-moe`
- ⚡ **Streaming SSE nativo**: Respuestas en tiempo real desde el LLM
- 🐳 **Desacoplado por capas**: LLM + Embeddings en el host (systemd), API en contenedor (Podman)
- 🚀 **Máximo rendimiento**: Binarios nativos con acceso directo a GPU
- 🔒 **Seguro**: Usuario no-root en contenedor, secrets fuera de la imagen
- 📦 **Moderno**: Gestión de paquetes con `uv`, Python 3.12, tipado estricto
- 🛡️ **HTTPS automático**: Caddy como proxy inverso con Let's Encrypt

## 🧠 ¿Qué es RAG y cómo funciona aquí?

**RAG (Retrieval-Augmented Generation)** es una técnica que combina la búsqueda de información relevante en una base de conocimientos con la generación de respuestas por un modelo de lenguaje.

### Flujo de una consulta con RAG

```
1. El usuario pregunta: "¿Cuándo son las inscripciones del periodo 1-2026?"

2. FastAPI convierte la pregunta en un vector (embedding) usando nomic-embed-text

3. ChromaDB busca los fragmentos de documentos más similares semánticamente

4. Se construye un "Súper Prompt" inyectando el contexto relevante:
   ┌─────────────────────────────────────────────────────────────┐
   │ System: Eres un asistente de la UNEFA Apure.              │
   │         Responde basándote en este contexto:              │
   │         [Fragmento del calendario académico 2026]         │
   │ User: ¿Cuándo son las inscripciones?                      │
   └─────────────────────────────────────────────────────────────┘

5. El LLM genera una respuesta precisa basada en TUS documentos
```

### Ventajas de usar RAG

- ✅ **Cero alucinaciones**: El LLM responde basado en documentos reales
- ✅ **Citas verificables**: Cada respuesta incluye la fuente del documento
- ✅ **Conocimiento actualizable**: Solo actualizas los documentos, sin reentrenar
- ✅ **Especialización**: El asistente sabe TODO sobre la UNEFA Apure

## 🏗️ Arquitectura

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
│  │ llama-server         │  │ llama-server         │            │
│  │ (systemd)            │  │ (systemd)            │            │
│  │ Puerto 8080          │  │ Puerto 8081          │            │
│  │ LLM principal        │  │ nomic-embed-text     │            │
│  │ (genera respuestas)  │  │ (genera embeddings)  │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### ¿Por qué esta arquitectura?

- **llama-server (LLM) en systemd**: Máximo rendimiento con acceso directo a GPU/CPU sin overhead de contenedor
- **llama-server (Embeddings) en systemd**: Servicio ligero (~350MB RAM) dedicado a vectorizar textos
- **FastAPI en Podman con --network=host**: Aislamiento del código Python, fácil despliegue, sin overhead de red
- **ChromaDB dentro del contenedor**: Base de datos vectorial persistente montada como volumen
- **Caddy como proxy inverso**: HTTPS automático, HTTP/3, y manejo de conexiones persistentes para streaming

## 📋 Requisitos previos

### En el servidor host

- **Linux** (probado en Debian/Ubuntu, Fedora)
- **Podman** >= 4.0
- **Caddy** >= 2.7
- **Python** 3.12+ (solo para desarrollo local y script de ingesta)
- **uv** >= 0.4 (gestor de paquetes)
- **llama.cpp** compilado o binario oficial descargado
- **GPU con drivers** (NVIDIA CUDA, AMD ROCm, o Vulkan) - opcional pero recomendado

### Modelos necesarios

1. **LLM principal** (ej: Qwen2.5, Llama 3, etc.) para generar respuestas
2. **nomic-embed-text-v2-moe.Q8_0.gguf** (~350MB) para generar embeddings

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
│   │   └── endpoints.py      # Rutas FastAPI (chat, models, health, rag/stats)
│   ├── core/
│   │   ├── config.py         # Configuración con pydantic-settings
│   │   └── schemas.py        # Modelos Pydantic (compatibles OpenAI)
│   ├── services/
│   │   ├── llm_service.py    # Cliente HTTP hacia llama-server (LLM)
│   │   └── rag_service.py    # Servicio RAG con ChromaDB + embeddings
│   └── main.py               # App FastAPI + middlewares
├── scripts/
│   └── ingesta_rag.py        # Script para poblar la base de conocimiento
├── Containerfile             # Imagen Podman optimizada con uv
├── pyproject.toml            # Dependencias del proyecto
├── uv.lock                   # Lockfile reproducible
├── .python-version           # 3.12
└── README.md
```

## ⚙️ Configuración

### Variables de entorno

Crea un archivo `.env` en la raíz del proyecto (ya está en `.gitignore`):

```env
# URL del llama-server (LLM principal) en el host
LLAMA_SERVER_URL=http://127.0.0.1:8080
LLAMA_TIMEOUT=300

# URL del llama-server (Embeddings) en el host
EMBEDDING_SERVER_URL=http://127.0.0.1:8081
EMBEDDING_MODEL=nomic-embed

# RAG - ChromaDB
CHROMA_PERSIST_DIR=/app/chroma_data
CHROMA_COLLECTION_NAME=unefa_knowledge
RAG_TOP_K=4

# API Key opcional para proteger el endpoint
API_KEY=tu-super-secreto-aqui

# CORS - dominios permitidos (cambia en producción)
CORS_ORIGINS=["https://tu-dominio.com"]
```

## 💻 Desarrollo local

### 1. Clonar e instalar

```bash
git clone https://github.com/BufferRing/pwa-virtual-assistant-backend.git
cd pwa-virtual-assistant-backend
uv sync
```

### 2. Iniciar llama-server (LLM principal) en una terminal

```bash
./llama-server \
    --model /ruta/a/tu/modelo.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 4096 \
    --n-gpu-layers 99
```

### 3. Iniciar llama-server (Embeddings) en otra terminal

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

La API estará en `http://127.0.0.1:8000` y la documentación interactiva en `http://127.0.0.1:8000/v1/openapi.json`.

### 6. Linting y tests

```bash
uv run ruff check .          # Linter
uv run ruff format .         # Formatter
uv run pytest                # Tests
```

## 🚀 Despliegue en producción

### Paso 1: Configurar llama-server (LLM) como servicio systemd

Crea el archivo `/etc/systemd/system/llama-server.service`:

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

### Paso 2: Configurar llama-server (Embeddings) como servicio systemd

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

### Paso 3: Activar ambos servicios

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo systemctl enable --now llama-embed

# Verificar que estén corriendo
sudo systemctl status llama-server
sudo systemctl status llama-embed
```

### Paso 4: Preparar la base de conocimiento

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

### Paso 5: Construir y ejecutar el contenedor de FastAPI

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

**Flags clave explicados:**
- `--env-file .env`: Inyecta las variables sin embeberlas en la imagen
- `--network=host`: El contenedor usa la red del host directamente (cero overhead de red)
- `-v /opt/chroma_data:/app/chroma_data:Z`: Monta la BD vectorial de forma persistente (la `:Z` es para SELinux en Fedora/RHEL)
- `--restart=always`: Reinicia si crashea

### Paso 6: Configurar Caddy

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

Envía mensajes al asistente. **Con RAG activo**, el sistema buscará automáticamente en la base de conocimiento el contexto relevante antes de responder.

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

Lista los modelos disponibles en llama-server.

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

### `GET /`

Endpoint raíz de verificación.

### Estrategia de Chunking

Los documentos se dividen en fragmentos (chunks) siguiendo estas reglas:

- **División por secciones (`##`)**: Mantiene la coherencia temática
- **Contexto heredado**: Cada chunk incluye los títulos padre para dar contexto
- **Tamaño óptimo**: 300-800 tokens por chunk
- **Overlap**: 10-15% de solapamiento entre chunks para no perder contexto

## 🐛 Troubleshooting

### El contenedor no puede conectar a llama-server

1. Verifica que ambos servicios estén corriendo:
   ```bash
   sudo systemctl status llama-server
   sudo systemctl status llama-embed
   ```

2. Prueba la conexión desde dentro del contenedor:
   ```bash
   podman exec -it unefa-app curl http://127.0.0.1:8080/health
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

Asegúrate de que tu Caddyfile tenga `flush_interval -1`. Sin esto, Caddy buferiza las respuestas SSE y el cliente no recibe los chunks en tiempo real.

### Error de permisos en ChromaDB (SELinux)

Si usas Fedora/RHEL y ves errores de permisos, asegúrate de montar el volumen con `:Z`:
```bash
-v /opt/chroma_data:/app/chroma_data:Z
```

### Ver logs

```bash
podman logs unefa-app
journalctl -u llama-server -f
journalctl -u llama-embed -f
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

| Componente | RAM aprox. | CPU |
|------------|-----------|-----|
| LLM (Qwen2.5-7B) | ~4-5 GB | Alto durante inferencia |
| Embeddings (Nomic) | ~350 MB | Bajo |
| FastAPI + ChromaDB | ~200-400 MB | Bajo |
| Caddy | ~50 MB | Bajo |

**Total recomendado**: Mínimo 8 GB RAM para operación fluida.

## 📄 Licencia

MIT © 2026 BufferRing. Ver [LICENSE](./LICENSE) para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir cambios importantes.

---

Hecho con ❤️ usando FastAPI, llama.cpp, ChromaDB, Podman y Caddy.

**Asistente especializado en la UNEFA Núcleo Apure** 🇻🇪
