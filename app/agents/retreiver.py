"""
retriver.py es como una interfaz mas avanzada que el agente va a utilizar para busqueda semantica
LangGraph lo va a utilizar para buscar contexto relevante antes de responder.

Por qué existe este archivo si ya tenemos VectorStore.search()?
    VectorStore.search() es la operación de bajo nivel: embedea la query,
    busca en PGVector, devuelve chunks crudos.
 
    Retriever agrega lógica de negocio encima:
        - Decide si el contexto recuperado es "suficiente" para responder
        - Formatea los chunks en un string de contexto listo para el LLM
        - Implementa la estrategia anti-alucinación #2: verificación de contexto
        - Agrega metadata útil para que el agente pueda citar fuentes
"""

import logging
from dataclasses import dataclass
 
from sqlalchemy.orm import Session
 
from app.config import config
from app.db.vectore_store import VectorStore
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class RetrievedChunk:
    """
    Un chunk recuperado de PGVector con su metadata.
    Usamos dataclass para tener type hints claros en el agente. O sea, le estamos diciendo que esto es un tipo 
    """
    id: str
    content: str
    source_filename: str
    chunk_index: str
    similarity: float
 
 
@dataclass
class RetrievalResult:
    """
    El resultado completo de una búsqueda semántica.
 
    El agente usa has_sufficient_context para decidir si responder
    o rechazar la pregunta con "no tengo suficiente información".
 
    context_text es el string listo para inyectar en el prompt del LLM.
    """
    chunks: list[RetrievedChunk]
    has_sufficient_context: bool
    context_text: str           # texto formateado listo para el prompt
    best_similarity: float      # score del chunk más relevante encontrado
    total_chunks_found: int     # cuántos chunks pasaron el filtro de similitud
 
class Retriever:
    """
    Interfaz de búsqueda semántica para el agente LangGraph.
    """
 
    def __init__(self, db: Session) -> None:
        self.db = db
        self.vector_store = VectorStore(db)
 
    async def retrieve(
        self,
        chat_id: str,
        query: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> RetrievalResult:
        """
        Busca los chunks más relevantes para una query en el chat dado.
 
        Estrategia anti-alucinación implementada aquí:
            Después de recuperar los chunks, verificamos que el mejor score
            de similitud supere el umbral mínimo. Si no lo supera, marcamos
            has_sufficient_context=False y el agente rechazará la pregunta.
 
            Esto es diferente de la estrategia #1 (en VectorStore.search):
                #1: filtra chunks individuales por debajo del umbral -> menos chunks
                #2: evalúa si el MEJOR chunk encontrado es suficientemente bueno
        """
        top_k = top_k or config.TOP_K_RESULTS
        min_similarity = min_similarity or config.MIN_SIMILARITY_THRESHOLD
 
        logger.info(
            f"Buscando contexto para chat_id={chat_id}, "
            f"query='{query[:80]}...', top_k={top_k}, min_sim={min_similarity}"
        )
 
        # Llamamos a VectorStore — aquí aplica la estrategia #1 (filtro por chunk)
        raw_chunks = await self.vector_store.search(
            chat_id=chat_id,
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
        )
 
        # Convertimos los dicts a dataclasses tipados
        chunks = [
            RetrievedChunk(
                id=c["id"],
                content=c["content"],
                source_filename=c["source_filename"],
                chunk_index=c["chunk_index"],
                similarity=c["similarity"],
            )
            for c in raw_chunks
        ]
 
        # Si no hay ningún chunk definitivamente sin contexto suficiente
        if not chunks:
            logger.info(
                f"No se encontraron chunks relevantes para chat_id={chat_id}. "
                f"Sin contexto suficiente."
            )
            return RetrievalResult(
                chunks=[],
                has_sufficient_context=False,
                context_text="",
                best_similarity=0.0,
                total_chunks_found=0,
            )
 
        best_similarity = chunks[0].similarity  # ya vienen ordenados desc por similitud
 
        # Estrategia anti-alucinación #2:
        # Verificamos que el mejor chunk sea genuinamente relevante.
        # Un score de 0.4 significa que hay correlación semántica real.
        # Menos de eso probablemente es ruido — el modelo encontró "algo"
        # pero no es realmente sobre el tema de la pregunta.
        has_sufficient_context = best_similarity >= min_similarity
 
        logger.info(
            f"Recuperados {len(chunks)} chunks. "
            f"Mejor similitud: {best_similarity:.4f}. "
            f"Contexto suficiente: {has_sufficient_context}."
        )
 
        # Formateamos el contexto como texto para inyectar en el prompt del LLM
        context_text = self._format_context(chunks) if has_sufficient_context else ""
 
        return RetrievalResult(
            chunks=chunks,
            has_sufficient_context=has_sufficient_context,
            context_text=context_text,
            best_similarity=best_similarity,
            total_chunks_found=len(chunks),
        )
 
    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Formatea los chunks como un bloque de contexto para el prompt del LLM.
 
        El formato incluye la fuente de cada chunk para que:
            1. El LLM pueda citar la fuente en su respuesta.
            2. Nosotros podamos mostrar "Basado en: capitulo3.pdf" en el frontend.
        """
        formatted_parts = []
 
        for chunk in chunks:
            part = (
                f"[Fuente: {chunk.source_filename}, fragmento {chunk.chunk_index}]\n"
                f"{chunk.content}"
            )
            formatted_parts.append(part)
 
        return "\n\n".join(formatted_parts)
 
    async def retrieve_for_generation(
        self,
        chat_id: str,
        query: str,
    ) -> tuple[str, list[str]]:
        """
        Versión simplificada del retriever para uso directo en nodos LangGraph.
 
        Devuelve una tupla (context_text, source_ids) donde:
            - context_text: string listo para inyectar en el prompt
            - source_ids:   lista de chunk IDs usados (para guardar en Message.source_chunks)
 
        Si no hay contexto suficiente, context_text será una cadena vacía
        y source_ids será una lista vacía — el nodo del agente debe verificar
        esto antes de llamar al LLM.
        """
        result = await self.retrieve(chat_id=chat_id, query=query)
 
        if not result.has_sufficient_context:
            return "", []
 
        source_ids = [chunk.id for chunk in result.chunks]
        return result.context_text, source_ids