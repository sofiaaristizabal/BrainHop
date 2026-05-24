"""
reject_node.py es el nodo que rechaza preguntas sin suficiente contexto.

Genera un mensaje de rechazo claro y amigable para el niño.

"""

import logging
from app.agents.state import ChatState

logger = logging.getLogger(__name__)

# Respuesta de rechazo estándar.
# Amigable, explica el problema, y da una acción concreta al usuario.
REJECTION_MESSAGE = (
    "Lo siento, no tengo suficiente información en mis documentos para responder esa pregunta. \n\n"
    "Puedes subir documentos relacionados con el tema (como apuntes de clase, "
    "capítulos del libro, o guías de estudio) y luego preguntarme de nuevo. "
    "¡Con más información podré ayudarte mejor!"
)


def reject_node(state: ChatState) -> ChatState:
    """
    Establece la respuesta de rechazo en el estado.

    Si hubo un error técnico (state["error"] está lleno), el mensaje
    es ligeramente diferente para no confundir al niño.
    """
    if state.get("error"):
        logger.error(
            f"[reject_node] Error técnico para chat_id={state['chat_id']}: {state['error']}"
        )
        response = (
            "Algo salió mal de mi lado "
            "Por favor intenta tu pregunta de nuevo en un momento."
        )
    else:
        logger.info(
            f"[reject_node] Rechazando pregunta por falta de contexto. "
            f"chat_id={state['chat_id']}, best_similarity={state.get('best_similarity', 0):.4f}"
        )
        response = REJECTION_MESSAGE

    return {
        **state,
        "agent_response": response,
    }