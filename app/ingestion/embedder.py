"""
embedder.py Genera embeddings con ollama y los guarda en PGVector
Recibe los chunks de texto ya divididos, los envia a ollama para obtener su representacion vectorial y los persiste en la base de datos 
También actualiza el document recor con el estado actual del rpocesamiento 
"""

import logging
from datetime import datetime, timezone
 
from sqlalchemy.orm import Session
 
from app.config import config
from app.db.models import DocumentRecord
from app.db.vectore_store import VectorStore
from app.ingestion.chunker import chunk_text, get_chunk_stats
from app.ingestion.loader import extract_text

logger = logging.getLogger(__name__)

#Ya con esto creamos el pipeline completo de ingesta
async def ingest_document(db:Session, document_record: DocumentRecord, file_bytes: bytes,)->DocumentRecord:
    """
    Ejecuta el pipeline completo de ingesta
    Pasos:
        1. Marca el documento como "processing"
        2. Extrae el texto según el tipo de archivo
        3. Divide el texto en chunks con overlap
        4. Genera embeddings para cada chunk via Ollama
        5. Guarda los embeddings en PGVector
        6. Marca el documento como "ready"
        7. Si algo falla → marca como "failed" con el mensaje de error
    Devuelve el DocumentRecord actualizado con el status final y chunk_count.
    Esta función hace commit de los cambios al DocumentRecord. El VectorStore hace flush de los chunks pero no commit 
    """

    # Paso 1: Marcar como processing 
    document_record.status = "processing"
    db.commit()
 
    logger.info(
        f"Iniciando ingesta del documento '{document_record.filename}' "
        f"(id={document_record.id}, chat_id={document_record.chat_id})"
    )
 
    try:
        # Paso 2: Extraer texto según el MIME type
        logger.info(f"Extrayendo texto de '{document_record.filename}'...")
        raw_text = extract_text(
            file_bytes=file_bytes,
            mime_type=document_record.file_type,
            filename=document_record.filename,
        )
        logger.info(f"Texto extraído: {len(raw_text)} caracteres.")
 
        # Paso 3: Dividir en chunks
        logger.info("Dividiendo texto en chunks...")
        chunks = chunk_text(raw_text)
        stats = get_chunk_stats(chunks)
 
        logger.info(
            f"Chunking completado: {stats['total_chunks']} chunks, "
            f"promedio {stats['avg_length']} chars/chunk."
        )
 
        if not chunks:
            raise ValueError(
                "El documento no produjo ningún chunk válido después del procesamiento. "
                "El archivo puede estar vacío o contener solo imágenes."
            )
 
        # Paso 4 y 5: Generar embeddings y guardar en PGVector
        # VectorStore llama a Ollama por cada chunk y hace flush al final.
        logger.info(f"Generando embeddings para {len(chunks)} chunks via Ollama...")
        vector_store = VectorStore(db)
        saved_count = await vector_store.save_chunks(
            chat_id=document_record.chat_id,
            document_id=document_record.id,
            source_filename=document_record.filename,
            chunks=chunks,
        )
 
        logger.info(f"Embeddings guardados: {saved_count} chunks en PGVector.")
 
        # Paso 6: Actualizar el DocumentRecord como exitoso
        document_record.status = "ready"
        document_record.chunk_count = saved_count
        document_record.processed_at = datetime.now(timezone.utc)
        document_record.error_message = None
 
        db.commit()
        db.refresh(document_record)
 
        logger.info(
            f"Ingesta completada exitosamente para '{document_record.filename}'. "
            f"Status: ready, chunks: {saved_count}."
        )
 
        return document_record
 
    except Exception as e:
        # Paso 7: Si algo salió mal, marcamos el documento como fallido
        # y guardamos el mensaje de error para debugging.
        db.rollback()
 
        error_msg = str(e)
        logger.error(
            f"Error durante la ingesta de '{document_record.filename}': {error_msg}",
            exc_info=True,  # esto incluye el stack trace completo en los logs
        )
 
        # Necesitamos una nueva transacción para actualizar el estado a "failed"
        try:
            document_record.status = "failed"
            document_record.error_message = error_msg[:1000]  # truncamos por si es muy largo
            db.add(document_record)
            db.commit()
            db.refresh(document_record)
        except Exception as db_error:
            # Si incluso esto falla, solo logueamos — no podemos hacer mucho más
            logger.error(f"Error adicional actualizando status a 'failed': {db_error}")
            db.rollback()
 
        return document_record
 
#Creamos una función que de una vez nos cree el document recor antes de ingestar 
def create_document_record(db: Session, chat_id: str, filename: str, file_type:str, file_size:int,) ->DocumentRecord:
    record = DocumentRecord(
        chat_id=chat_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        status="pending",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
 
    logger.info(
        f"DocumentRecord creado: id={record.id}, "
        f"filename='{filename}', chat_id={chat_id}"
    )
 
    return record