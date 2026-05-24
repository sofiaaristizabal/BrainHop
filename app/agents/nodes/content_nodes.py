"""
content_nodes.py contiene todos los nodos que generan el contenido educativo tras la ingesta.

Estos tres nodos corren en PARALELO en el grafo de generación de contenido:
    - summarizer_node: crea las 10 flashcards del tema
    - keywords_node: da 5 conceptos clave con definiciones para el juego que crearemos 
    - quiz_node: crea 3 preguntas de selección múltiple para un quiz

Todos leen del mismo campo `full_context` del estado y escriben
en campos distintos, por eso pueden correr en paralelo sin conflictos.

Cada nodo pide al LLM responder en JSON puro — parseamos y validamos
el resultado con Pydantic antes de guardarlo. Si el LLM devuelve
JSON malformado, reintentamos una vez antes de fallar.
"""

import json
import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.llm import get_llm
from app.agents.state import ContentGenerationState

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> str:
    """
    Extrae JSON de la respuesta del LLM aunque venga envuelto en markdown.
    El LLM a veces responde: ```json\n[...]\n```
    Esta función extrae solo el JSON puro.
    """
    # Intentar extraer bloque ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # Si no hay bloque, asumir que todo el texto es JSON
    return text.strip()


async def _call_llm_for_json(system_prompt: str, user_prompt: str) -> str:
    """Llama al LLM y retorna el JSON extraído como string"""
    llm = get_llm(temperature=0.4)  # más temperatura para contenido más variado
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    return _extract_json(response.content)



# Context loader: primer nodo del grafo de generación, Carga el contexto del chat en un solo string para que los nodos paralelos lo tengan disponible.
async def context_loader_node(state: ContentGenerationState) -> ContentGenerationState:
    """
    Recupera todos los chunks del chat y los concatena en un contexto completo.
    Este nodo corre ANTES de los tres nodos paralelos.
    """
    from langchain_ollama import OllamaEmbeddings
    from langchain_postgres.vectorstores import PGVector
    from app.config import config

    logger.info(f"[context_loader] Cargando contexto para chat_id={state['chat_id']}")

    try:
        embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=state["chat_id"],
            connection=config.DATABASE_URL,
            use_jsonb=True,
        )

        # Para generación de contenido queremos MÁS contexto que para Q&A, traemos los 20 chunks más representativos del material
        docs = vector_store.similarity_search(
            query="resumen general del tema principal",
            k=20,
        )

        if not docs:
            return {**state, "full_context": "", "error": "No hay documentos en este chat."}

        context_parts = [doc.page_content for doc in docs]
        full_context = "\n\n".join(context_parts)

        logger.info(f"[context_loader] Contexto cargado: {len(full_context)} chars, {len(docs)} chunks")

        return {**state, "full_context": full_context}

    except Exception as e:
        logger.error(f"[context_loader] Error: {e}", exc_info=True)
        return {**state, "full_context": "", "error": str(e)}

# Summarizer — genera 10 flashcards

SUMMARIZER_SYSTEM = """Eres un asistente educativo experto en crear materiales de estudio para niños.
Tu tarea es crear EXACTAMENTE 10 tarjetas de estudio (flashcards) basadas en el texto proporcionado.

IMPORTANTE: Responde ÚNICAMENTE con un array JSON válido. Sin texto adicional, sin explicaciones, sin comillas de código markdown.

Formato exacto requerido:
[
  {"title": "Título corto del concepto", "content": "Explicación simple en 2-3 oraciones"},
  ...
]

Reglas:
- Cada tarjeta debe cubrir UN concepto diferente e importante del texto
- El título debe tener máximo 5 palabras
- El contenido debe ser fácil de entender para un niño de 10-14 años
- Usa lenguaje simple y directo
- EXACTAMENTE 10 tarjetas, ni más ni menos"""

SUMMARIZER_USER = """Crea 10 flashcards de estudio basadas en este texto:

{context}

Recuerda: responde SOLO con el array JSON, sin ningún texto adicional."""


async def summarizer_node(state: ContentGenerationState) -> ContentGenerationState:
    """Genera 10 flashcards del contenido del chat."""
    logger.info(f"[summarizer_node] Generando flashcards para chat_id={state['chat_id']}")

    if not state.get("full_context"):
        return {**state, "flashcards": []}

    try:
        json_str = await _call_llm_for_json(
            system_prompt=SUMMARIZER_SYSTEM,
            user_prompt=SUMMARIZER_USER.format(context=state["full_context"][:6000]),
        )

        flashcards = json.loads(json_str)

        # Validación básica de estructura
        validated = []
        for card in flashcards:
            if isinstance(card, dict) and "title" in card and "content" in card:
                validated.append({
                    "title": str(card["title"])[:100],
                    "content": str(card["content"])[:500],
                })

        logger.info(f"[summarizer_node] {len(validated)} flashcards generadas")
        return {**state, "flashcards": validated}

    except json.JSONDecodeError as e:
        logger.error(f"[summarizer_node] JSON inválido del LLM: {e}")
        return {**state, "flashcards": []}
    except Exception as e:
        logger.error(f"[summarizer_node] Error: {e}", exc_info=True)
        return {**state, "flashcards": []}


# Keywords : genera 5 conceptos clave con definiciones

KEYWORDS_SYSTEM = """Eres un asistente educativo. Tu tarea es identificar los 5 conceptos más importantes del texto dado.

IMPORTANTE: Responde ÚNICAMENTE con un array JSON válido. Sin texto adicional.

Formato exacto:
[
  {"term": "Nombre del concepto", "definition": "Definición simple en 1-2 oraciones"},
  ...
]

Reglas:
- Elige los conceptos más fundamentales para entender el tema
- Las definiciones deben ser claras para un niño de 10-14 años
- EXACTAMENTE 5 conceptos"""

KEYWORDS_USER = """Identifica los 5 conceptos clave de este texto:

{context}

Responde SOLO con el array JSON."""


async def keywords_node(state: ContentGenerationState) -> ContentGenerationState:
    """Genera 5 conceptos clave con definiciones."""
    logger.info(f"[keywords_node] Generando keywords para chat_id={state['chat_id']}")

    if not state.get("full_context"):
        return {**state, "keywords": []}

    try:
        json_str = await _call_llm_for_json(
            system_prompt=KEYWORDS_SYSTEM,
            user_prompt=KEYWORDS_USER.format(context=state["full_context"][:6000]),
        )

        keywords = json.loads(json_str)

        validated = []
        for kw in keywords:
            if isinstance(kw, dict) and "term" in kw and "definition" in kw:
                validated.append({
                    "term": str(kw["term"])[:100],
                    "definition": str(kw["definition"])[:300],
                })

        logger.info(f"[keywords_node] {len(validated)} keywords generados")
        return {**state, "keywords": validated}

    except json.JSONDecodeError as e:
        logger.error(f"[keywords_node] JSON inválido: {e}")
        return {**state, "keywords": []}
    except Exception as e:
        logger.error(f"[keywords_node] Error: {e}", exc_info=True)
        return {**state, "keywords": []}

# Quiz — genera 3 preguntas de selección múltiple

QUIZ_SYSTEM = """Eres un asistente educativo. Crea EXACTAMENTE 3 preguntas de selección múltiple sobre el texto dado.

IMPORTANTE: Responde ÚNICAMENTE con un array JSON válido. Sin texto adicional.

Formato exacto:
[
  {
    "question": "¿Texto de la pregunta?",
    "options": [
      {"label": "A", "text": "Primera opción"},
      {"label": "B", "text": "Segunda opción"},
      {"label": "C", "text": "Tercera opción"},
      {"label": "D", "text": "Cuarta opción"}
    ],
    "correct_answer": "A",
    "explanation": "Breve explicación de por qué esta es la respuesta correcta"
  },
  ...
]

Reglas:
- Las preguntas deben estar basadas ÚNICAMENTE en el texto
- Cada pregunta debe tener exactamente 4 opciones (A, B, C, D)
- Solo una opción es correcta
- Las opciones incorrectas deben ser plausibles (no obviamente falsas)
- Lenguaje apropiado para niños de 10-14 años
- EXACTAMENTE 3 preguntas"""

QUIZ_USER = """Crea 3 preguntas de selección múltiple sobre este texto:

{context}

Responde SOLO con el array JSON."""


async def quiz_node(state: ContentGenerationState) -> ContentGenerationState:
    """Genera 3 preguntas de selección múltiple."""
    logger.info(f"[quiz_node] Generando quiz para chat_id={state['chat_id']}")

    if not state.get("full_context"):
        return {**state, "quiz": []}

    try:
        json_str = await _call_llm_for_json(
            system_prompt=QUIZ_SYSTEM,
            user_prompt=QUIZ_USER.format(context=state["full_context"][:6000]),
        )

        questions = json.loads(json_str)

        validated = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            if not all(k in q for k in ("question", "options", "correct_answer", "explanation")):
                continue
            if not isinstance(q["options"], list) or len(q["options"]) != 4:
                continue

            validated.append({
                "question": str(q["question"])[:500],
                "options": [
                    {"label": opt["label"], "text": str(opt["text"])[:200]}
                    for opt in q["options"]
                    if isinstance(opt, dict) and "label" in opt and "text" in opt
                ],
                "correct_answer": str(q["correct_answer"]).upper(),
                "explanation": str(q["explanation"])[:300],
            })

        logger.info(f"[quiz_node] {len(validated)} preguntas generadas")
        return {**state, "quiz": validated}

    except json.JSONDecodeError as e:
        logger.error(f"[quiz_node] JSON inválido: {e}")
        return {**state, "quiz": []}
    except Exception as e:
        logger.error(f"[quiz_node] Error: {e}", exc_info=True)
        return {**state, "quiz": []}


# Content saver — último nodo, guarda todo en Chat.generated_content

def content_saver_node(state: ContentGenerationState, db) -> ContentGenerationState:
    """
    Guarda el contenido generado en la base de datos.
    Igual que history_node, recibe db vía closure en graph.py.
    """
    from app.db.models import Chat
    from datetime import datetime, timezone

    logger.info(f"[content_saver] Guardando contenido para chat_id={state['chat_id']}")

    try:
        chat = db.get(Chat, state["chat_id"])
        if not chat:
            logger.error(f"[content_saver] Chat {state['chat_id']} no encontrado")
            return {**state, "error": "Chat no encontrado"}

        chat.generated_content = {
            "flashcards": state.get("flashcards", []),
            "keywords": state.get("keywords", []),
            "quiz": state.get("quiz", []),
        }
        chat.is_ready = True
        chat.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(chat)

        logger.info(f"[content_saver] Contenido guardado para chat_id={state['chat_id']}")
        return state

    except Exception as e:
        logger.error(f"[content_saver] Error: {e}", exc_info=True)
        db.rollback()
        return {**state, "error": str(e)}