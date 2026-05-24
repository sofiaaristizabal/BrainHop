"""
guard_node.py es el nodo que decide si hay suficiente contexto para responder.

Este nodo es una arista condicional en el grafo: si has_sufficient_context
es False, el grafo va al nodo de rechazo. Si es True, va al generador.

No llama al LLM es una decisión puramente lógica basada en los scores
de similitud calculados por el retriever. Rápido y sin costo computacional.
"""

import logging
from app.agents.state import ChatState

logger = logging.getLogger(__name__)


def guard_node(state: ChatState) -> ChatState:
    """
    Verifica si el contexto recuperado es suficiente para responder.

    Este nodo no modifica el estado — solo lo lee. La lógica de routing
    la hace la función route_after_guard() que se usa como arista
    condicional en el grafo.
    """
    logger.info(
        f"[guard_node] chat_id={state['chat_id']} "
        f"has_sufficient_context={state['has_sufficient_context']} "
        f"best_similarity={state.get('best_similarity', 0):.4f}"
    )
    return state


def route_after_guard(state: ChatState) -> str:
    """
    Función de routing para la arista condicional después del guard.

    Retorna el nombre del nodo al que debe ir el grafo:
        "generator"  — hay contexto suficiente, generamos respuesta
        "reject"     — no hay contexto, rechazamos la pregunta

    """
    if state.get("error"):
        return "reject"

    if state["has_sufficient_context"]:
        return "generator"

    return "reject"