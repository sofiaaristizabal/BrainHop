"""
vectorstore.py se encarga de guardar document embedings usando PGVector

    WRITE (Ingestion):
        Recibe una lista de chunks, genera un embedding para cada uno de ellos usando ollama y guarda su el embedding y su metadata en la tabla de embeddings 
    READ:
        recibe la pregunta del usuario, genera un embedding para ella a traves de  ollama, encuentra los chunks con el mayor semantic similarity y los devuelve como contexto para que el LLM responda
    Isolation model:
        a cada chunk se le asigna un chat_id para saber a que onversación pertenece 
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import Column, DateTime, Float, Index, String, Text, delete, select
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session

from pgvector.sqlalchemy import Vector

from app.config import config
from app.db import Base 

#Creamos la tabla de embeddings

class EmbeddingChunk(Base):
    """
    Una fila equivale a un chunk de texto mas su vector embedding

     A single uploaded document is split into many chunks (e.g. 512 tokens each
    with 50-token overlap). Each chunk gets its own row here. When the agent
    searches for relevant context it computes the cosine distance between the
    query embedding and every chunk embedding for the current chat_id, then
    returns the top-k closest chunks.

    cada chunk pertenece a un chat especifico porque los documentos son de cada chat
    """

    __tablename__="embedding_chunks"

    id: Column = Column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )

    chat_id: Column = Column(
        String(36), 
        nullable=False, 
        index=True
    )

    document_id: Column = Column(
        String(36),
          nullable=False, 
          index=True
    )

    source_filename: Column = Column( # The original filename — stored here so the agent can cite the source
        String(500), 
        nullable=False
    )

    content: Column = Column( # The actual text content of this chunk. Returned alongside the embedding
        Text, 
        nullable=False
    )

    #Posición del chunk en el documento de dpnde proviene 
    chunk_index: Column = Column(
        String(10),
        nullable=False
    )

    #La representación numerica del content, su dimensión ya la establecimos en el config.py 
    embedding: Column = Column(
        Vector(config.EMBEDDING_DIM), 
        nullable= False
    )

    created_at: Column = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

"""
Imagine you have a huge physical library containing thousands of textbooks. If a kid asks your app a question about their biology homework, the backend has to look through every single sentence in the library to find the right answer. That takes way too long.

An Index is like the Index/Glossary at the back of a textbook, or a cheat-sheet map for PostgreSQL. Instead of reading the whole database from scratch every time, the database checks the index map to jump straight to the exact rows it needs.

When a kid opens "Chat #5", your API runs a query saying: "Hey database, find me only the text chunks where chat_id is equal to 5."

Without these indexes: PostgreSQL has to do a "Full Table Scan." It reads row 1, row 2, row 3, all the way to row 1,000,000 to find the chunks for Chat #5. It’s painfully slow.

With these indexes: PostgreSQL creates a tiny, ultra-fast lookup tree behind the scenes. It instantly goes, "Ah, Chat #5? Those are rows 450 through 460." It skips reading the rest of the database entirely.
"""

Index("ix_embedding_chunks_chat_id", EmbeddingChunk.chat_id)

Index("ix_embedding_chunks_document_id", EmbeddingChunk.document_id)

async def get_embedding(text: str) -> list[float]:
    