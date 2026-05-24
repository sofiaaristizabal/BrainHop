"""
history_node.py es el nodo que persiste la conversación en la base de datos.

Último nodo del grafo de chat. Guarda tanto el mensaje del usuario
como la respuesta del agente en la tabla messages.

guardamos aqui porque si el LLM falla, no queremos guardar un mensaje de usuario sin respuesta. Guardamos el par completo o ninguno.
"""

import logging
from sqlalchemy.orm import Session
from app.agents.state import ChatState
from app.db.models import Message

logger = logging.getLogger(__name__)


def history_node(state: ChatState, db: Session) -> ChatState:
    """
    Guarda el mensaje del usuario y la respuesta del agente en la DB.

    Nota sobre la inyección de db:
        LangGraph no tiene DI nativa como FastAPI. Pasamos la sesión
        usando un closure en graph.py:
            node = lambda state: history_node(state, db)
        Así cada request tiene su propia sesión de DB.
    """
    logger.info(f"[history_node] Guardando conversación para chat_id={state['chat_id']}")

    try:
        # Mensaje del usuario
        user_msg = Message(
            chat_id=state["chat_id"],
            role="user",
            content=state["question"],
            source_chunks=None,
            retrieval_score=None,
        )
        db.add(user_msg)

        # Mensaje del agente  incluye los chunks usados y el score
        agent_msg = Message(
            chat_id=state["chat_id"],
            role="agent",
            content=state["agent_response"],
            source_chunks=state.get("source_chunk_ids") or [],
            retrieval_score=state.get("best_similarity"),
        )
        db.add(agent_msg)

        db.commit()
        db.refresh(user_msg)
        db.refresh(agent_msg)

        logger.info(
            f"[history_node] Mensajes guardados: "
            f"user={user_msg.id}, agent={agent_msg.id}"
        )

        return {
            **state,
            "user_message_id": user_msg.id,
            "agent_message_id": agent_msg.id,
        }

    except Exception as e:
        logger.error(f"[history_node] Error guardando en DB: {e}", exc_info=True)
        db.rollback()
        # No fallamos el grafo por esto — el usuario ya tiene su respuesta
        return {
            **state,
            "user_message_id": None,
            "agent_message_id": None,
            "error": str(e),
        }