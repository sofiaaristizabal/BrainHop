"""
chat_service.py es la bussines logic para los chats
"""

from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.db.models import Chat, User
from app.api.schemas.schemas import ChatCreate

from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from app.config import config

def create_chat(db: Session, data: ChatCreate, user: User) -> Chat:
    """Creamos un nuevo chat para el usuario"""
    chat = Chat(
        user_id = user.id,
        title = data.title,
        topic = data.topic
    )
    db.add(chat)
    db.commit()
    db.refresh(chat) #hacemos un refresh para traer el id que se creo automiticamente, el is_ready que por defecto  es false, el cretaed_at que se genera automaticamente, etc 
    return chat


def get_chat(db: Session, chat_id: str, user: User):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat con id {chat_id} no encontrado")
    
    if chat.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Acceso denegado.")
    
    return chat

def get_user_chats(db: Session, user: User) -> list[Chat]:
    chats = db.query(Chat).filter(Chat.user_id == user.id).order_by(Chat.updated_at.desc()).all()


def delete_chat(db:Session, chat_id: str, user:User) ->None:
    """Eliminamos un chat y todos los datos asociados al chat"""
    chat = get_chat(db, chat_id, User)

    try:
        embeddings = OllamaEmbeddings(
            model = config.EMBEDDING_MODEL,
            base_url= config.OLLAMA_URL
        )
        
        vectore_store = PGVector(
            embeddings= embeddings,
            collection_name = chat_id,
            connection = config.DATABASE_URL,
            use_jsonb=True
        )

        vectore_store.delete_collection()
    except Exception:
        pass

    db.delete(chat)
    db.commit()

