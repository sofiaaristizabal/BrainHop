"""
retriever_node.py es el nodo que recupera contexto relevante de PGVector.

Es el primer nodo del grafo de chat. Toma la pregunta del usuario,
busca los chunks más similares en la base vectorial del chat,
y llena el estado con el contexto y los scores de similitud.
"""

import logging
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector

from app.agents.state import ChatState
from app.config import config

logger = logging.getLogger(__name__)


async def retriever_node(state: ChatState) -> ChatState:
    """
    Busca contexto relevante para la pregunta del usuario.

    Usa LangChain's PGVector con collection_name=chat_id para que
    cada chat tenga su propio espacio de búsqueda aislado.

    Llena en el estado:
        retrieved_context     — texto formateado con fuentes
        source_chunk_ids      — IDs de los chunks encontrados
        best_similarity       — score del chunk más relevante
        has_sufficient_context — False si nada supera el umbral
    """
    logger.info(f"[retriever_node] chat_id={state['chat_id']} query='{state['question'][:60]}...'")

    try:
        embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

        # collection_name=chat_id garantiza aislamiento entre chats
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=state["chat_id"],
            connection=config.DATABASE_URL,
            use_jsonb=True,
        )

        # similarity_search_with_relevance_scores devuelve (Document, score) donde score es similitud coseno (1.0 = idéntico, 0.0 = sin relación)
        results = vector_store.similarity_search_with_relevance_scores(
            query=state["question"],
            k=config.TOP_K_RESULTS,
        )

        if not results:
            logger.info(f"[retriever_node] Sin resultados para chat_id={state['chat_id']}")
            return {
                **state,
                "retrieved_context": "",
                "source_chunk_ids": [],
                "best_similarity": 0.0,
                "has_sufficient_context": False,
            }

        # Filtramos por umbral mínimo — estrategia anti-alucinación #1
        relevant = [
            (doc, score)
            for doc, score in results
            if score >= config.MIN_SIMILARITY_THRESHOLD
        ]

        best_similarity = results[0][1]  # el primero siempre es el más similar

        # Estrategia anti-alucinación #2: verificamos que el MEJOR chunk sea bueno
        has_sufficient_context = (
            len(relevant) > 0 and best_similarity >= config.MIN_SIMILARITY_THRESHOLD
        )

        logger.info(
            f"[retriever_node] {len(relevant)}/{len(results)} chunks relevantes. "
            f"Mejor similitud: {best_similarity:.4f}. "
            f"Suficiente: {has_sufficient_context}"
        )

        if not has_sufficient_context:
            return {
                **state, #ake all the key-value pairs currently inside the state dictionary, unpack them one by one, and copy them right here into this brand-new dictionary."
                "retrieved_context": "",
                "source_chunk_ids": [],
                "best_similarity": best_similarity,
                "has_sufficient_context": False,
            }

        # Formateamos el contexto con fuentes para el prompt
        context_parts = []
        source_ids = []

        for doc, score in relevant:
            source = doc.metadata.get("source", "documento desconocido")
            context_parts.append(f"[Fuente: {source}]\n{doc.page_content}")
            # LangChain guarda el ID en metadata si lo pusiste al ingestar
            if doc_id := doc.metadata.get("id"):
                source_ids.append(doc_id)

        context_text = "\n\n".join(context_parts)

        return {
            **state,
            "retrieved_context": context_text,
            "source_chunk_ids": source_ids,
            "best_similarity": best_similarity,
            "has_sufficient_context": True,
        }

    except Exception as e:
        logger.error(f"[retriever_node] Error: {e}", exc_info=True)
        return {
            **state,
            "retrieved_context": "",
            "source_chunk_ids": [],
            "best_similarity": 0.0,
            "has_sufficient_context": False,
            "error": str(e),
        }