"""
chunker.py — Divide Documents de LangChain en chunks usando RecursiveCharacterTextSplitter.

Recibe la lista de Documents que devuelve loader.py y los divide en
fragmentos mas pequenos con overlap, usando el splitter de LangChain.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import config


def chunk_documents( documents: list[Document],  chunk_size: int | None = None, chunk_overlap: int | None = None,) -> list[Document]:
    """
    Divide una lista de Documents en chunks con overlap.

    El metadata de cada Document original (source, page) se copia
    automaticamente a cada chunk que se derive de el. Asi, cuando el
    agente recupera un chunk, sabe de que archivo y pagina viene.

    devuelve la lista de Documents divididos, cada uno con su metadata original.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        # Separadores jerarquicos: de lo mas semantico a lo mas granular
        separators=["\n\n", "\n", ". ", " ", ""], #parrafos, lineas, oraciones, palabras y caracteres
        add_start_index=True,  # add_start_index guarda la posicion del chunk dentro del documento
    )

    chunks = splitter.split_documents(documents)

    # Filtramos chunks demasiado cortos para ser utiles
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 50]

    return chunks


def get_chunk_stats(chunks: list[Document]) -> dict:
    """
    Estadisticas sobre los chunks generados.
    """
    if not chunks:
        return {"total_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0, "total_chars": 0}

    lengths = [len(c.page_content) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": sum(lengths),
    }