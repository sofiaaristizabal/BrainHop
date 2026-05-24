"""
models.py es un SQLAlchemy ORM, aqui definimos los modelos (las entidades que vamos a tener en la base de datos relacional, o sea toda la estructura de la parte no vectorial)
 - define la estructura de cada tabla dentro de brainhop_db
 - Cada chat pertenee a un usuario
 - Cada chat tiene multiples mensajes, documentos, resultdos de quizes, etc 
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, Float, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

#generar un nuevo UUID como valor por defecto para llaves primarias
def new_uuid()->str:
    return str(uuid.uuid4())

def now_utc()-> datetime:
    return datetime.now(timezone.utc)

class User(Base):

    __tablename__="users"

    id: Mapped[str] = mapped_column( # Mapped[...] is a Type Hint introduced in SQLAlchemy 2.0. It bridges the gap between your Python code editor and your database columns. Mapped[str] tells your IDE: "Hey, when I pull a row from the database and touch this field in Python, treating it like a standard Python str string is completely safe.
        UUID(as_uuid=False),
        primary_key=True,
        default=new_uuid
    )

    keycloak_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False
        )
    
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False
    )
    username: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=now_utc,
        nullable=False
    )

    chats: Mapped[list["Chat"]] = relationship(
        "Chat",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"

class Chat(Base):

    __tablename__="chats"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=new_uuid
    )

    #Foreign key
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Chat"
    )

    topic: Mapped[str | None] = mapped_column(
        String(255),
          nullable=True)
    
    #mira si los documentos han sido procesados o no
    is_ready: Mapped[bool] = mapped_column(
        Boolean, 
        default=False, 
        nullable=False
    )

    generated_content: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=now_utc, 
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=now_utc, 
        onupdate=now_utc, 
        nullable=False
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="chats"
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )

    documents: Mapped[list["DocumentRecord"]] = relationship(
        "DocumentRecord", 
        back_populates="chat", 
        cascade="all, delete-orphan"
    )

    quiz_results: Mapped[list["QuizResult"]] = relationship(
        "QuizResult", 
        back_populates="chat", 
        cascade="all, delete-orphan"
    )
 
    def __repr__(self) -> str:
        return f"<Chat id={self.id} title={self.title} user_id={self.user_id}>"
    
class Message(Base):
    __tablename__="messages"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True,
        default=new_uuid
    )

    #Foreign key
    chat_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False
    )

    #If the message was from the user or the agent
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    #Which documents chunks were used to generate this reponse, se guarda como una lista de JSONs del chunk id
    source_chunks: Mapped[list |None] = mapped_column(
        JSON,
        nullable=True
    )

    #Similaruty score
    retreival_source: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=now_utc, 
        nullable=False
    )

    chat: Mapped["Chat"] = relationship(
        "Chat",
        back_populates="messages"
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role} chat_id={self.chat_id}>"
    
class DocumentRecord(Base):
    __tablename__="document_records"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=new_uuid
    )

    #Foreign key
    chat_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False
    )

    filename: Mapped[str] = mapped_column(
        String(500), nullable=False
    )

    #MIME type
    file_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    file_size:Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    #status: si el archivo esta pending, processing, ready o si esta failed
    status:Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False
    )

    #en cuantos chunks fue dividido el archivo
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=True
    )

    # If status == "failed", we store the error message here for debugging.
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=now_utc, 
        nullable=False
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), 
        nullable=True
    )

    chat: Mapped["Chat"] = relationship(
        "Chat",
        back_populates="documents"
    )

    def __repr__(self) -> str:
        return f"<DocumentRecord id={self.id} filename={self.filename} status={self.status}>"
    

class QuizResult(Base):

    __tablename__ = "quiz_results"
 
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=new_uuid
    )
    chat_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), 
        ForeignKey("chats.id", ondelete="CASCADE"), 
        nullable=False
    )
 
    #Respuestas correctas
    score: Mapped[int] = mapped_column(
        Integer, 
        nullable=False
    )

    total_questions: Mapped[int] = mapped_column(
        Integer, 
        nullable=False
    )
 
    ## {"question_index": 0, "selected": "B", "correct": "B", "is_correct": true}
    answers: Mapped[list | None] = mapped_column(
        JSON, 
        nullable=True
    )
 
    taken_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=now_utc, nullable=False
    )
 
    chat: Mapped["Chat"] = relationship(
        "Chat",
        back_populates="quiz_results"
    )
 
    def __repr__(self) -> str:
        return f"<QuizResult id={self.id} score={self.score}/{self.total_questions}>"

    


