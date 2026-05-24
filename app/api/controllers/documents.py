"""
documents.py es el enpoint para la ingesta y destión de documentos
 
GET    /api/documents/chat/{chat_id} : lista documentos de un chat
POST   /api/documents/chat/{chat_id} : subir documento (ingesta en background)
GET    /api/documents/{document_id} : estado de un documento (para polling)
DELETE /api/documents/{document_id} : eliminar documento y sus embeddings
"""

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, File
from sqlalchemy.orm import Session
 
from app.api.deps import get_db, get_current_user
from app.db.models import User, DocumentRecord
from app.api.schemas.schemas import DocumentResponse, SuccessResponse
from app.api.services.document_service import (
    get_chat_documents,
    upload_document,
    delete_document,
)
from app.api.services.chat_service import get_chat
from fastapi import HTTPException, status
 
router = APIRouter()
 
 
@router.get("/chat/{chat_id}", response_model=list[DocumentResponse])
def list_documents(chat_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),):
    """Lista todos los documentos subidos a un chat."""
    return get_chat_documents(db, chat_id, current_user)
 
 
@router.post("/chat/{chat_id}", response_model=DocumentResponse, status_code=202)
async def upload_document_endpoint( chat_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user),):
    """
    Sube un documento y dispara la ingesta en background.
    Retorna 202 Accepted con el DocumentRecord en status="pending".
    """
    return await upload_document(db, chat_id, file, current_user, background_tasks)
 
 
@router.get("/{document_id}", response_model=DocumentResponse)
def get_document_status( document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user), ):
    """
    Retorna el estado actual de un documento.
    """
    record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Documento no encontrado.")
    # Verificar ownership
    get_chat(db, record.chat_id, current_user)
    return record
 
 
@router.delete("/{document_id}", response_model=SuccessResponse)
def delete_document_endpoint(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),):
    """Elimina un documento y todos sus embeddings de PGVector."""
    delete_document(db, document_id, current_user)
    return {"message": "Documento eliminado correctamente."}
 