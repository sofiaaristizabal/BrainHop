"""
routes/messages.py — Endpoints de conversación y quiz.

GET  /api/messages/chat/{chat_id}: historial de mensajes
POST /api/messages/chat/{chat_id}/ask : hacer una pregunta al agente
POST /api/messages/chat/{chat_id}/quiz : enviar respuestas del quiz
GET  /api/messages/chat/{chat_id}/quiz : historial de resultados del quiz
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.db.models import User, QuizResult
from app.api.schemas.schemas import (
    AskResponse,
    MessageCreate,
    MessageResponse,
    QuizSubmit,
    QuizResultResponse,
)
from app.api.services.message_service import (
    get_chat_history,
    ask_question,
    submit_quiz,
)
from app.api.services.chat_service import get_chat

router = APIRouter()


@router.get("/chat/{chat_id}", response_model=list[MessageResponse])
def get_history(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retorna el historial completo de mensajes de un chat."""
    return get_chat_history(db, chat_id, current_user)


@router.post("/chat/{chat_id}/ask", response_model=AskResponse)
async def ask(
    chat_id: str,
    body: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Envía una pregunta al agente y retorna la respuesta.
    Retorna tanto el mensaje del usuario como el del agente en una sola llamada.
    """
    return await ask_question(db, chat_id, body.content, current_user)


@router.post("/chat/{chat_id}/quiz", response_model=QuizResultResponse, status_code=201)
def submit_quiz_endpoint(
    chat_id: str,
    body: QuizSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Envía las respuestas del quiz y retorna el resultado con score."""
    return submit_quiz(db, chat_id, body, current_user)


@router.get("/chat/{chat_id}/quiz", response_model=list[QuizResultResponse])
def get_quiz_history(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retorna el historial de intentos del quiz para un chat."""
    get_chat(db, chat_id, current_user)
    return (
        db.query(QuizResult)
        .filter(QuizResult.chat_id == chat_id)
        .order_by(QuizResult.taken_at.desc())
        .all()
    )