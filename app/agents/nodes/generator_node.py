"""
generator_node.py es el nodo que genera la respuesta usando el LLM y el contexto RAG.

Solo llega aquí si el guard confirmó que hay contexto suficiente.
Construye un prompt que obliga al LLM a responder ÚNICAMENTE basándose
en el contexto recuperado nunca inventa información.
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.llm import get_llm
from app.agents.state import ChatState

logger = logging.getLogger(__name__)

# Creamos el prompt del sistema
# Escrito en español porque los niños interactúan en español, porque la app la pense como para  Colombia 
# El {context} y {question} se reemplazan en tiempo de ejecución.
SYSTEM_PROMPT = """Eres BrainHop, un asistente educativo amigable diseñado para ayudar a niños a aprender.

Tu trabajo es responder preguntas basándote ÚNICAMENTE en el siguiente contexto extraído de los documentos del estudiante.

REGLAS ESTRICTAS:
1. Responde SOLO con información del contexto. Si la respuesta no está en el contexto, dilo claramente.
2. Usa lenguaje simple y claro, apropiado para niños de primaria o secundaria.
3. Si el contexto menciona la fuente, puedes citarla.
4. Sé alentador y positivo en tu tono.
5. Responde en el mismo idioma que la pregunta del estudiante.
6. NUNCA inventes información que no esté en el contexto.
7. Puedes usar analogías que un niño entenderia y que le ayudarian a asociar conceptos ya que los cerebros de los niños entienden mejor asi.
8. No des respuestas extremadamente largas, los niños tienen un spam de atencion no muy largo. 

CONTEXTO DE LOS DOCUMENTOS:
{context}"""

USER_PROMPT = """Pregunta del estudiante: {question}

Responde basándote únicamente en el contexto proporcionado."""


async def generator_node(state: ChatState) -> ChatState:
    """
    Llama al LLM con el contexto recuperado para generar una respuesta.
   crea el agent_response 
    """
    logger.info(f"[generator_node] Generando respuesta para chat_id={state['chat_id']}")

    try:
        llm = get_llm(temperature=0.2)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(
                context=state["retrieved_context"]
            )),
            HumanMessage(content=USER_PROMPT.format(
                question=state["question"]
            )),
        ]

        response = await llm.ainvoke(messages)
        agent_response = response.content.strip()

        logger.info(
            f"[generator_node] Respuesta generada: {len(agent_response)} chars"
        )

        return {
            **state,
            "agent_response": agent_response,
        }

    except Exception as e:
        logger.error(f"[generator_node] Error: {e}", exc_info=True)
        return {
            **state,
            "agent_response": "Lo siento, tuve un problema al generar la respuesta. Por favor intenta de nuevo.",
            "error": str(e),
        }