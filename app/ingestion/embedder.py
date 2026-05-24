"""
embedder.py  Genera embeddings con ollama y los guarda en PGVector. Orquesta el pipeline completo de ingesta usando LangChain.
Recibe los chunks de texto ya divididos, los envia a ollama para obtener su representacion vectorial y los persiste en la base de datos 
También actualiza el document recor con el estado actual del rpocesamiento 

Pasos:
    1. load_documents() : extrae texto con el loader correcto
    2. chunk_documents() : divide con RecursiveCharacterTextSplitter
    3. PGVector.add_documents() : genera embeddings con Ollama y los guarda

"""

import logging
import uuid
from datetime import datetime, timezone

from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy.orm import Session

from app.config import config
from app.db.models import DocumentRecord
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents, get_chunk_stats

logger = logging.getLogger(__name__)


def _get_vector_store(chat_id: str) -> PGVector:
    """
    Construye la instancia de PGVector para un chat especifico.

    collection_name=chat_id es la clave del aislamiento: cada chat
    tiene su propio espacio de busqueda en PGVector.
    """
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=chat_id,
        connection=config.DATABASE_URL,
        use_jsonb=True,
    )


async def ingest_document( db: Session, document_record: DocumentRecord,  file_bytes: bytes,) -> DocumentRecord:
    """
    Ejecuta el pipeline completo de ingesta para un documento.

    Pasos:
        1. Marca el documento como "processing"
        2. Carga el texto con LangChain loaders
        3. Divide en chunks con RecursiveCharacterTextSplitter
        4. Genera embeddings con Ollama y guarda en PGVector
        5. Marca el documento como "ready"
        6. Si algo falla, marca como "failed" con el mensaje de error
    """
    # Paso 1: marcar como processing 
    document_record.status = "processing"
    db.commit()

    logger.info(
        f"Iniciando ingesta: '{document_record.filename}' "
        f"(id={document_record.id}, chat_id={document_record.chat_id})"
    )

    try:
        # Paso 2: cargar el texto
        logger.info(f"Cargando texto de '{document_record.filename}'...")
        documents = load_documents(
            file_bytes=file_bytes,
            mime_type=document_record.file_type,
            filename=document_record.filename,
        )
        logger.info(f"Cargados {len(documents)} documentos/paginas.")

        # Paso 3: dividir en chunks
        logger.info("Dividiendo en chunks...")
        chunks = chunk_documents(documents)
        stats = get_chunk_stats(chunks)
        logger.info(
            f"Chunking completado: {stats['total_chunks']} chunks, "
            f"promedio {stats['avg_length']} chars/chunk."
        )

        if not chunks:
            raise ValueError(
                "El documento no produjo chunks validos. "
                "Puede estar vacio o contener solo imagenes."
            )

        # Agregamos el document_id al metadata de cada chunk para poder
        # borrar todos los chunks de un documento cuando el usuario lo elimine.
        for chunk in chunks:
            chunk.metadata["document_id"] = document_record.id

        # Paso 4: generar embeddings y guardar en PGVector
        # PGVector.add_documents() llama a Ollama internamente por cada chunk.
        logger.info(f"Generando embeddings para {len(chunks)} chunks via Ollama...")
        vector_store = _get_vector_store(document_record.chat_id)

        # Generamos IDs unicos para cada chunk para poder referenciarlos despues
        ids = [str(uuid.uuid4()) for _ in chunks]
        vector_store.add_documents(documents=chunks, ids=ids) #aqui estamos haciendo el embedding y pasandolo a la base de datos como vector

        logger.info(f"Embeddings guardados: {len(chunks)} chunks en PGVector.")

        # Paso 5: actualizar el DocumentRecord como exitoso
        document_record.status = "ready"
        document_record.chunk_count = len(chunks)
        document_record.processed_at = datetime.now(timezone.utc)
        document_record.error_message = None
        db.commit()
        db.refresh(document_record)

        logger.info(f"Ingesta completada: '{document_record.filename}' -> ready.")
        return document_record

    except Exception as e:
        db.rollback()
        error_msg = str(e)
        logger.error(f"Error en ingesta de '{document_record.filename}': {error_msg}", exc_info=True)

        # Nueva transaccion para marcar como failed
        try:
            document_record.status = "failed"
            document_record.error_message = error_msg[:1000]
            db.add(document_record)
            db.commit()
            db.refresh(document_record)
        except Exception as db_error:
            logger.error(f"Error adicional marcando failed: {db_error}")
            db.rollback()

        return document_record

#Creamos una función que de una vez nos cree el document recor antes de ingestar 
def create_document_record( db: Session, chat_id: str, filename: str, file_type: str, file_size: int,) -> DocumentRecord:

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
    logger.info(f"DocumentRecord creado: id={record.id}, filename='{filename}'")
    return record