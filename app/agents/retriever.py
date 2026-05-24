"""
retriever.py  Busqueda semantica usando LangChain PGVector.

retriver.py es como una interfaz mas avanzada que el agente va a utilizar para busqueda semantica
LangGraph lo va a utilizar para buscar contexto relevante antes de responder.

Por qué existe este archivo si ya tenemos VectorStore.search()?
    VectorStore.search() es la operación de bajo nivel: embedea la query,
    busca en PGVector, devuelve chunks crudos.

El agente usa esto para encontrar contexto relevante antes de responder.
Reemplaza la implementacion manual anterior con LangChain's PGVector,
que maneja internamente el embedding de la query y la busqueda por similitud.
"""

import logging
from dataclasses import dataclass

from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector

from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Resultado completo de una busqueda semantica.

    El agente usa has_sufficient_context para decidir si responder
    o rechazar. context_text es el string listo para inyectar en el prompt.
    """
    has_sufficient_context: bool
    context_text: str         # texto formateado con fuentes, listo para el prompt
    source_ids: list[str]     # IDs de los chunks usados (para Message.source_chunks)
    best_similarity: float    # score del chunk mas relevante (0.0-1.0)
    total_found: int          # cuantos chunks pasaron el filtro


class Retriever:
    """
    Interfaz de busqueda semantica para los nodos LangGraph.

    """

    def _get_vector_store(self, chat_id: str) -> PGVector:
        """Construye PGVector para el chat especifico."""
        embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        return PGVector(
            embeddings=embeddings,
            collection_name=chat_id,   # aislamiento por chat
            connection=config.DATABASE_URL,
            use_jsonb=True,
        )

    async def retrieve(
        self,
        chat_id: str,
        query: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> RetrievalResult:
        """
        Busca los chunks mas relevantes para una query.

        Estrategia anti-alucinacion #1: filtramos chunks con similitud
        por debajo del umbral minimo.
        Estrategia anti-alucinacion #2: verificamos que el MEJOR chunk
        encontrado sea suficientemente bueno antes de responder.
        """
        top_k = top_k or config.TOP_K_RESULTS
        min_similarity = min_similarity or config.MIN_SIMILARITY_THRESHOLD

        logger.info(
            f"[Retriever] chat_id={chat_id}, "
            f"query='{query[:60]}...', top_k={top_k}, min_sim={min_similarity}"
        )

        try:
            vector_store = self._get_vector_store(chat_id)

            # similarity_search_with_relevance_scores retorna (Document, score) donde score es similitud coseno: 1.0=identico, 0.0=sin relacion
            results = vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
            )

            if not results:
                logger.info(f"[Retriever] Sin resultados para chat_id={chat_id}")
                return RetrievalResult(
                    has_sufficient_context=False,
                    context_text="",
                    source_ids=[],
                    best_similarity=0.0,
                    total_found=0,
                )

            best_similarity = results[0][1]

            # Estrategia #1: filtrar chunks individuales por debajo del umbral
            relevant = [(doc, score) for doc, score in results if score >= min_similarity]

            # Estrategia #2: verificar que el mejor chunk sea genuinamente bueno
            has_sufficient_context = len(relevant) > 0 and best_similarity >= min_similarity

            logger.info(
                f"[Retriever] {len(relevant)}/{len(results)} chunks relevantes. "
                f"Mejor: {best_similarity:.4f}. Suficiente: {has_sufficient_context}"
            )

            if not has_sufficient_context:
                return RetrievalResult(
                    has_sufficient_context=False,
                    context_text="",
                    source_ids=[],
                    best_similarity=best_similarity,
                    total_found=0,
                )

            # Formateamos el contexto con etiquetas de fuente para que el LLM sepa de donde viene cada fragmento y pueda citarlo
            context_parts = []
            source_ids = []

            for doc, score in relevant:
                source = doc.metadata.get("source", "documento desconocido")
                page = doc.metadata.get("page", "")
                page_info = f", pagina {page}" if page != "" else ""
                context_parts.append(
                    f"[Fuente: {source}{page_info}]\n{doc.page_content}"
                )
                if doc_id := doc.metadata.get("id"):
                    source_ids.append(doc_id)

            context_text = "\n\n".join(context_parts)

            return RetrievalResult(
                has_sufficient_context=True,
                context_text=context_text,
                source_ids=source_ids,
                best_similarity=best_similarity,
                total_found=len(relevant),
            )

        except Exception as e:
            logger.error(f"[Retriever] Error: {e}", exc_info=True)
            return RetrievalResult(
                has_sufficient_context=False,
                context_text="",
                source_ids=[],
                best_similarity=0.0,
                total_found=0,
            )