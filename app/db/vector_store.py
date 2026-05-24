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
from sqlalchemy.orm import Mapped, mapped_column, relationship, Session

from pgvector.sqlalchemy import Vector

from app.config import config
from app.db.base import Base 

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

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )

    chat_id: Mapped[str]= mapped_column(
        String(36), 
        nullable=False, 
        index=True
    )

    document_id: Mapped[str]= mapped_column(
        String(36),
          nullable=False, 
          index=True
    )

    source_filename:Mapped[str]= mapped_column( # The original filename — stored here so the agent can cite the source
        String(500), 
        nullable=False
    )

    content: Mapped[str]= mapped_column( # The actual text content of this chunk. Returned alongside the embedding
        Text, 
        nullable=False
    )

    #Posición del chunk en el documento de dpnde proviene 
    chunk_index: Mapped[str]= mapped_column(
        String(10),
        nullable=False
    )

    #La representación numerica del content, su dimensión ya la establecimos en el config.py 
    embedding: Column = Column(
        Vector(config.EMBEDDING_DIM), 
        nullable= False
    )

    created_at: Mapped[datetime]= mapped_column(
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

#Private helper function, la utilizaremos en la ingesta para convertir texto a embeddings 
async def _get_embedding(text: str) -> list[float]:
    """
    Llama a la instancia de docker de ollama usando su URL para generar vector embedding para text
    toma texto plano (como un parrafo o la pregunta de un niño), lo envia al nomic embeded text model de ollama el cual lo convierte en una lista de decimales (un vector) 
    """

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{config.OLLAMA_URL}/api/embeddings",
            json = {
                "model": config.EMBEDDING_MODEL,
                "prompt": text,
            },
        )

        response.raise_for_status()
        data = response.json()

    embedding: list[float] = data["embedding"]

    if len(embedding) != config.EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {config.EMBEDDING_DIM}, "
            f"got {len(embedding)}. Check EMBEDDING_MODEL and EMBEDDING_DIM in config."
        )
    
    return embedding

class Vectorestore:
    """
    Interfaz para guardar y hacer querys de los documentos
    la usamos durante la ingesta para guardar los chunks en la base de datos, para buscar por similitud y traer los resultados y para eliminar chunks
    """

    def __init__(self, db: Session) -> None:
        self.db = db # recivimos la sesión de la base de datos por inyección de dependencias
    

    #WRITE

    async def save_chunks(
            self,
            chat_id: str,
            document_id: str,
            source_filename:str,
            chunks: list[str],
    ) -> int:
        """
        Insertamos y especificamos una lista de text chunks para un chat y documento especifico

        Este metodo es llamado por el ingestion pipeline despues de que un documento ya fue dividido en chunks. Cdasa chunk es ingestado individulmente y guardado en una fila de la db
        El metodo devuelve el numero de chunks guardados exitosamente 
        """

        saved_count = 0

        for index, chunk_text in enumerate(chunks):
            if not chunk_text.strip():# Skip empty chunks that sometimes appear after splitting.
                continue
            embedding = await _get_embedding(chunk_text)
            chunk_record = EmbeddingChunk( #Creamos el objeto ORM y lo añadimos a la sesión
                chat_id=chat_id,
                document_id=document_id,
                source_filename=source_filename,
                content=chunk_text,
                chunk_index=str(index),
                embedding=embedding,
            )
            self.db.add(chunk_record)
            saved_count +=1
        
        self.db.flush() # Flush sends the INSERT statements to Postgres within the current transaction, but does NOT commit. The caller commits after also  updating the DocumentRecord.status to "ready".

        return saved_count
    
    async def search(
        self,
        chat_id: str,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """
        Vamos a buscar los chunks mas relevantes para las preguntas de los usuarios 
        - le hacemos embedding a la query con el mismo modelo que se uso para la ingesta
        - El computado calcula la distancia coseno entre la query y todos los chunks embedded en el chat history
        - devuelve el top_k (5) chunks mas cercamos filtrados por similitud (el mas relevante primero)
        """

        query_embedding = await _get_embedding(query)

        distance_col = EmbeddingChunk.embedding.cosine_distance(query_embedding).label("distance")

        stmt = (
            select(
                EmbeddingChunk.id,
                EmbeddingChunk.content,
                EmbeddingChunk.source_filename,
                EmbeddingChunk.chunk_index,
                distance_col,
            )
            .where(EmbeddingChunk.chat_id == chat_id)
            .order_by(distance_col)
            .limit(top_k)
        ) 

        rows = self.db.execute(stmt).fetchall()

        #convertimos la distancia a similitud y filtramos segun el threshol que habiamos definido en config
        results = []

        for row in rows:
            similarity = 1.0 -(row.distance/ 2.0) # Convert cosine distance (0–2) to similarity score (0–1).
            if similarity < min_similarity:
                continue

            results.append(
                {
                    "id": row.id,
                    "content": row.content,
                    "source_filename": row.source_filename,
                    "chunk_index": row.chunk_index,
                    "similarity": round(similarity, 4),
                }
            )
        return results
        
    def delete_document_chunks(self, document_id: str) -> int: 
        """
        Eliminar todos los embeded chunks asociados a un documento
        Lo llamamos cuando un usuario elimina un documento
        """

        stmt = delete(EmbeddingChunk).where(
            EmbeddingChunk.document_id == document_id
        )

        result = self.db.execute(stmt)
        self.db.flush()
        return result.rowcount
    
    def delete_chat_chunks(self, chat_id: str) -> int:
        """
        Eliminamos todos los embedded chunks de un chat cuando el usuario elimina el chat 
        """

        stmt = delete(EmbeddingChunk).where(EmbeddingChunk.chat_id == chat_id)
        result = self.db.execute(stmt)
        self.db.flush()
        return result.rowcount
    
    def get_chunk_count(self, chat_id: str) -> int:
        """Retornamos el total de chunks en un chat"""
        stmt = select(EmbeddingChunk).where(EmbeddingChunk.chat_id == chat_id)
        return len(self.db.execute(stmt).fetchall())
 
    def get_document_chunks(self, document_id: str) -> list[EmbeddingChunk]:
        """Retornamos todos los chunks de un documento especifico"""
        stmt = select(EmbeddingChunk).where(
            EmbeddingChunk.document_id == document_id
        ).order_by(EmbeddingChunk.chunk_index)
        return self.db.execute(stmt).scalars().all()