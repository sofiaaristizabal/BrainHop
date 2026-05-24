"""
llm.py  Instancia compartida del LLM de Ollama.

Un solo lugar donde se construye el LLM. Todos los nodos importan de aquí.
"""

from langchain_ollama import ChatOllama
from app.config import config

def get_llm(temperature: float = 0.3) -> ChatOllama:
    """
    Construye y retorna una instancia del LLM de Ollama.

    temperature=0.3 es un buen default para tareas educativas:
        - Lo suficientemente bajo para respuestas consistentes y precisas
        - Lo suficientemente alto para que no suene robótico
    Para generación de contenido (flashcards, quiz) usamos 0.5 para
    más variedad. Para el guard node usamos 0.0 para decisiones binarias.
    """
    return ChatOllama(
        model=config.CHAT_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
    )