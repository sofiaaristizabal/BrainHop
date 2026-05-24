"""
schemas.py son los DTOs de la aplicación usando Pydantic.
 
- Validamos los datos que entran por la API (request bodies)
- Serializamos los datos que salen (response bodies)
- Desacoplan el contrato de la API de los modelos de la DB
"""
 
from datetime import datetime
from typing import Any
from pydantic import BaseModel, EmailStr, Field
 

# User
class UserResponse(BaseModel):
    id: str
    keycloak_id: str
    email: EmailStr
    username: str
    is_active: bool
    created_at: datetime
 
    model_config = {"from_attributes": True}  # permite crear desde ORM objects dding from_attributes = True acts as a translator. It tells Pydantic:"If you are handed an object that isn't a dictionary, don't crash. Try reading its fields as attributes directly using dot notation instead!"This allows your FastAPI backend to safely take a database query result and send it directly out of the API.
 

# Chat
class ChatCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=255)
    topic: str | None = Field(default=None, max_length=255)
 
 
class ChatResponse(BaseModel):
    id: str
    user_id: str
    title: str
    topic: str | None
    is_ready: bool
    generated_content: dict | None  # flashcards, keywords, quiz
    created_at: datetime
    updated_at: datetime
 
    model_config = {"from_attributes": True}
 
 
class ChatListResponse(BaseModel):
    """Lista resumida de chats para el sidebar — sin generated_content (es pesado)"""
    id: str
    title: str
    topic: str | None
    is_ready: bool
    created_at: datetime
    updated_at: datetime
 
    model_config = {"from_attributes": True}
 
# Document
class DocumentResponse(BaseModel):
    id: str
    chat_id: str
    filename: str
    file_type: str
    file_size: int
    status: str           # pending | processing | ready | failed
    chunk_count: int | None
    error_message: str | None
    uploaded_at: datetime
    processed_at: datetime | None
 
    model_config = {"from_attributes": True}
 

# Message / Chat Q&A
class MessageCreate(BaseModel):
    """El cliente envía solo la pregunta. El chat_id viene del path parameter."""
    content: str = Field(..., min_length=1, max_length=5000)
 
 
class MessageResponse(BaseModel):
    id: str
    chat_id: str
    role: str             # "user" | "agent"
    content: str
    source_chunks: list[str] | None  # IDs de chunks usados
    retrieval_score: float | None    # mejor similitud encontrada
    created_at: datetime
 
    model_config = {"from_attributes": True}
 
 
class AskResponse(BaseModel):
    """
    Respuesta completa a una pregunta del usuario.
    Incluye tanto el mensaje del usuario como la respuesta del agente,
    así el frontend puede renderizar ambos en la misma llamada.
    """
    user_message: MessageResponse
    agent_message: MessageResponse
 
 

# Generated content (flashcards, keywords, quiz) 

class Flashcard(BaseModel):
    title: str
    content: str
 
 
class Keyword(BaseModel):
    term: str
    definition: str
 
 
class QuizOption(BaseModel):
    label: str   # "A", "B", "C", "D"
    text: str
 
 
class QuizQuestion(BaseModel):
    question: str
    options: list[QuizOption]
    correct_answer: str   # "A", "B", "C", or "D"
    explanation: str      # por qué esa es la respuesta correcta
 
 
class GeneratedContent(BaseModel):
    """
    Estructura del campo Chat.generated_content.
    El agente llena esto después de ingestar documentos.
    """
    flashcards: list[Flashcard]
    keywords: list[Keyword]
    quiz: list[QuizQuestion]
 
# Quiz result
 
class QuizAnswer(BaseModel):
    question_index: int
    selected: str   # "A", "B", "C", or "D"
 
 
class QuizSubmit(BaseModel):
    answers: list[QuizAnswer]
 
 
class QuizResultResponse(BaseModel):
    id: str
    chat_id: str
    score: int
    total_questions: int
    answers: list[dict] | None
    taken_at: datetime
 
    model_config = {"from_attributes": True}
 
# Generic responses

class SuccessResponse(BaseModel):
    message: str
 
 
class ErrorResponse(BaseModel):
    detail: str