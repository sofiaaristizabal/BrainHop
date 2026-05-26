"""
document_service.py es la logica de negocio de los documentos
"""

import logging
from fastapi import HTTPException, UploadFile, status, BackgroundTasks
from sqlalchemy.orm import Session
import asyncio
from app.config import config
from app.db.models import DocumentRecord, User
from app.ingestion.embedder import ingest_document, create_document_record
from app.api.services.chat_service import get_chat
from app.db.models import Chat
from app.agents.graph import build_content_graph
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector

logger = logging.getLogger(__name__)

def get_chat_documents(db: Session, chat_id: str, user: User) -> list[DocumentRecord]:
    """Retorna todos los documentos de un chat verificando ownership."""
    get_chat(db, chat_id, user)  # verifica que el chat existe y pertenece al user
    return (
        db.query(DocumentRecord)
        .filter(DocumentRecord.chat_id == chat_id)
        .order_by(DocumentRecord.uploaded_at.desc())
        .all()
    )

async def upload_document(db:Session, chat_id:str, file: UploadFile, user:User, background_task:BackgroundTasks) ->DocumentRecord:
    """
    Validamos el archivo, creamos el document record, disparamos la ingesta de documentos como una background task
    """
    logger.info(f"entramos al metodo")
    #Verificamos primero que el chat exista
    get_chat(db, chat_id, user)

    #Validamos el MIME type
    if file.content_type not in config.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Tipo de archivo no soportado: '{file.content_type}'. "
                f"Tipos permitidos: {config.ALLOWED_MIME_TYPES}"
            ),
        )
    
    file_bytes = await file.read()

    #Validamos el tamaño
    if len(file_bytes) > config.MAX_UPLOAD_SIZE_BYTES:
        max_mb = config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Archivo demasiado grande. Máximo permitido: {max_mb} MB.",
        )
    
    #Creamos el record con status "pending"
    record = create_document_record(
        db=db,
        chat_id=chat_id,
        filename=file.filename,
        file_type=file.content_type,
        file_size=len(file_bytes),
    )

    #Disparamos la ingesta como un background task 
    background_task.add_task(
        _run_ingestion_and_generate_content,
        db=db,
        record=record,
        file_bytes=file_bytes,
        chat_id=chat_id,
    )

    return record

def _run_ingestion_and_generate_content(db:Session, record: DocumentRecord, file_bytes: bytes, chat_id: str) ->None:
    """Coremos la ingesta completa y disparamos la generación de contenido, o sea, disparamos un flujo de langGraph"""

    updated_record = ingest_document(db=db, document_record=record, file_bytes=file_bytes)
 
    if updated_record.status == "ready":
        # Verificar si ya hay contenido generado para este chat
        chat = db.get(Chat, chat_id)
        if chat and not chat.is_ready:
            logger.info(f"Disparando generación de contenido para chat_id={chat_id}")
            _trigger_content_generation(db=db, chat_id=chat_id)

def _trigger_content_generation(db: Session, chat_id: str) -> None:
    """Corre el grafo de generación de contenido (flashcards, keywords, quiz)."""
    chat = db.get(Chat, chat_id)
    if not chat:
        return
 
    try:
        graph = build_content_graph(db)
        asyncio.run(graph.ainvoke({
            "chat_id": chat_id,
            "user_id": chat.user_id,
            "full_context": "",
            "flashcards": [],
            "keywords": [],
            "quiz": [],
            "error": None,
        }))
        logger.info(f"Contenido generado para chat_id={chat_id}")
        chat.is_ready = True
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error generando contenido para chat_id={chat_id}: {e}", exc_info=True)


def delete_document(db: Session, document_id: str, user: User) -> None:

    record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Documento no encontrado.")
 
    # Verificar ownership a través del chat
    get_chat(db, record.chat_id, user)
 
    # Borrar los chunks de este documento de PGVector
    # LangChain PGVector guarda el document_id en metadata — lo usamos para filtrar
    try:
        embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=record.chat_id,
            connection=config.DATABASE_URL,
            use_jsonb=True,
        )
        # Borramos todos los chunks donde metadata.document_id == document_id
        vector_store.delete(filter={"document_id": record.id})
    except Exception as e:
        logger.warning(f"No se pudieron borrar embeddings del documento {document_id}: {e}")
 
    db.delete(record)
    db.commit()