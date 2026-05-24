"""
graph.py ensambla los nodos en grafos LangGraph.

Define dos grafos:
    1. chat_graph: para Q&A en tiempo real
    2. content_generation_graph: para generar flashcards/keywords/quiz tras la ingesta

"""

from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session

from app.agents.state import ChatState, ContentGenerationState
from app.agents.nodes.retriever_node import retriever_node
from app.agents.nodes.guard_node import guard_node, route_after_guard
from app.agents.nodes.generator_node import generator_node
from app.agents.nodes.reject_node import reject_node
from app.agents.nodes.history_node import history_node
from app.agents.nodes.content_nodes import (
    context_loader_node,
    summarizer_node,
    keywords_node,
    quiz_node,
    content_saver_node,
)

# Grafo 1: Chat Q&A
# Flujo:
#   retriever -> guard -> [generator | reject] -> history -> END
#
# retriever: busca contexto en PGVector
# guard:     decide si hay suficiente contexto (arista condicional)
# generator: llama al LLM con contexto (si hay suficiente)
# reject:    responde "no tengo información" (si no hay suficiente)
# history:   guarda ambos mensajes en la DB
#

def build_chat_graph(db: Session) -> StateGraph:
    """
    Construye el grafo de chat con la sesión de DB inyectada via closure.

    Llamar esto UNA VEZ por request HTTP, no globalmente, para que
    cada request tenga su propia sesión de DB.
    """
    graph = StateGraph(ChatState)

    # Registramos los nodos
    # Los nodos que necesitan DB los envolvemos en lambdas (closure pattern)
    graph.add_node("retriever", retriever_node)
    graph.add_node("guard", guard_node)
    graph.add_node("generator", generator_node)
    graph.add_node("reject", reject_node)
    graph.add_node("history", lambda state: history_node(state, db))

    # Definimos el flujo
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "guard")

    # Arista condicional: el guard decide hacia dónde ir
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "generator": "generator",
            "reject": "reject",
        }
    )

    # Ambas ramas (generator y reject) terminan en history
    graph.add_edge("generator", "history")
    graph.add_edge("reject", "history")
    graph.add_edge("history", END)

    return graph.compile()


# Grafo 2: Generación de contenido educativo
#
# Flujo:
#   context_loader -> [summarizer | keywords | quiz] (paralelo) -> content_saver -> END
#
# context_loader: carga el contexto completo del chat
# summarizer, keywords, quiz: corren en PARALELO (LangGraph los ejecuta
#   simultáneamente cuando todos parten del mismo nodo)
# content_saver: espera a los tres y guarda en Chat.generated_content
#

def build_content_graph(db: Session) -> StateGraph:
    """
    Construye el grafo de generación de contenido.

    Los tres nodos de generación (summarizer, keywords, quiz) corren
    en paralelo porque en LangGraph, cuando múltiples nodos tienen
    aristas desde el mismo nodo fuente, se ejecutan concurrentemente.
    """
    graph = StateGraph(ContentGenerationState)

    graph.add_node("context_loader", context_loader_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("keywords", keywords_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("content_saver", lambda state: content_saver_node(state, db))

    graph.set_entry_point("context_loader")

    # Fan-out: context_loader → los tres nodos en paralelo
    graph.add_edge("context_loader", "summarizer")
    graph.add_edge("context_loader", "keywords")
    graph.add_edge("context_loader", "quiz")

    # Fan-in: los tres nodos convergen en content_saver
    graph.add_edge("summarizer", "content_saver")
    graph.add_edge("keywords", "content_saver")
    graph.add_edge("quiz", "content_saver")

    graph.add_edge("content_saver", END)

    return graph.compile()