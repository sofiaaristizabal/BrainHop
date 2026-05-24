"""
state.py define los estados de los grafos LangGraph.

En LangGraph, el "estado" es un diccionario tipado que fluye entre nodos.
Cada nodo recibe el estado completo, modifica algunos campos, y lo devuelve.
Los campos que un nodo no toca permanecen igual.

Tenemos dos grafos con dos estados distintos:

   - ContentGenerationState para el grafo que genera flashcards/keywords/quiz
    tras la ingesta de documentos.
   - ChatState para el grafo que maneja preguntas del usuario en tiempo real.
"""

from typing import Any
from typing_extensions import TypedDict

#Corre cuando se ingestan los documentos al chat
class ContentGenerationState(TypedDict):
    # Inputs (se setean antes de correr el grafo) 
    chat_id: str
    user_id: str

    # El contexto completo recuperado de todos los documentos del chat, el nodo context_loader lo llena; los nodos de generación lo leen.
    full_context: str

    #  Outputs (cada nodo de generación llena su campo) 
    flashcards: list[dict]    # [{title, content}, ...]
    keywords: list[dict]      # [{term, definition}, ...]
    quiz: list[dict]          # [{question, options, correct_answer, explanation}, ...]

    #Control Si algo falla, el nodo de error lo llena y el grafo termina.
    error: str | None

# Corre en cada pregunta del usuario.
class ChatState(TypedDict):
    # Inputs 
    chat_id: str
    user_id: str
    question: str             # la pregunta del usuario tal como la escribió

    # El nodo retriever llena estos campos.
    retrieved_context: str    # texto formateado con fuentes, listo para el prompt
    source_chunk_ids: list[str]   # IDs de los chunks usados (para Message.source_chunks)
    best_similarity: float    # mejor score de similitud encontrado (0.0–1.0)
    has_sufficient_context: bool  # el nodo guard manda al rechazo

    # El nodo generator o el nodo reject llena estos.
    agent_response: str       # la respuesta final que se muestra al usuario

    # El nodo history_saver llena estos después de guardar en DB.
    user_message_id: str | None
    agent_message_id: str | None

    error: str | None