"""
chats.py es el CRUD de chats
GET    /api/chats : lista de chats del usuario
POST   /api/chats : crear nuevo chat
GET    /api/chats/{id} : detalle de un chat (incluye generated_content)
DELETE /api/chats/{id}  : borrar chat y todos sus datos
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.db.models import User
from app.api.schemas.schemas import ChatCreate, ChatListResponse, ChatResponse, SuccessResponse
from app.api.services.chat_service import get_user_chats, get_chat, create_chat, delete_chat

router = APIRouter()

@router.get("", response_model= list[ChatListResponse])
def list_chats(db:Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Lista todos los chats de un usuario"""
    return get_user_chats(db, current_user)

@router.post("", response_model=ChatResponse, status_code=201)
def create_new_chat(data: ChatCreate, db:Session = Depends(get_db),  current_user: User = Depends(get_current_user)):
    return create_chat(db, data, current_user)

@router.get("/{chat_id}", response_model=ChatResponse)
def get_chat_detail(chat_id:str, db:Session = Depends(get_db),   current_user: User = Depends(get_current_user)):
    return get_chat(db, chat_id, current_user)

@router.delete("/{chat_id}", response_model=SuccessResponse)
def delete_chat_endpoint(chat_id:str, db:Session = Depends(get_db),  current_user: User = Depends(get_current_user)):
    delete_chat(db, chat_id, current_user)
    return {"message": "Chat eliminado correctamente "}