"""
message_service.py es la logica del chatbot como tal
"""

import logging
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
 
from app.agents.graph import build_chat_graph
from app.agents.state import ChatState
from app.db.models import Message, Chat, QuizResult, User
from app.api.schemas.schemas import AskResponse, MessageResponse, QuizSubmit, QuizResultResponse
from app.api.services.chat_service import get_chat

 
logger = logging.getLogger(__name__)

def get_chat_history(db: Session, chat_id: str, user: User) -> list[Message]:
    """Retorna el historial de mensajes de un chat en orden cronológico."""
    get_chat(db, chat_id, user) #devuelve un error si el chat no pertenece al usuario
    return (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc())
        .all()
    )

async def ask_question(db:Session, chat_id:str, question:str, user:User)-> AskResponse:
    """
    Procesa la pregunta del usuario a través del grafo LangGraph. Verificamos que el chat exista
    contruimos el grafo, invocamos el grafo, recuperamos los mensajes guardados, retornamos tanto pregunta como respuesta
    """

    chat = get_chat(db, chat_id, user)
    if not chat.is_ready:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Este chat no tiene documentos procesados todavía. Sube un documento primero.",
        )
    
    initial_state: ChatState = {
        "chat_id": chat_id,
        "user_id": user.id,
        "question": question,
        "retrieved_context": "",
        "source_chunk_ids": [],
        "best_similarity": 0.0,
        "has_sufficient_context": False,
        "agent_response": "",
        "user_message_id": None,
        "agent_message_id": None,
        "error": None,
    }

    try:
        # Construimos el grafo con la sesión actual (closure pattern)
        graph = build_chat_graph(db)
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Error en el grafo de chat para chat_id={chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno procesando tu pregunta. Intenta de nuevo.",
        )
    
    # Recuperar los mensajes guardados por history_node
    user_msg_id = final_state.get("user_message_id")
    agent_msg_id = final_state.get("agent_message_id")
 
    if not user_msg_id or not agent_msg_id:
        # El grafo corrió pero history_node falló al guardar
        # Construimos los mensajes desde el estado para no perder la respuesta
        logger.warning(f"history_node no guardó mensajes para chat_id={chat_id}")
        return AskResponse(
            user_message=MessageResponse(
                id="temp",
                chat_id=chat_id,
                role="user",
                content=question,
                source_chunks=None,
                retrieval_score=None,
                created_at=None,
            ),
            agent_message=MessageResponse(
                id="temp",
                chat_id=chat_id,
                role="agent",
                content=final_state.get("agent_response", "Error al procesar."),
                source_chunks=final_state.get("source_chunk_ids"),
                retrieval_score=final_state.get("best_similarity"),
                created_at=None,
            ),
        )
 
    user_msg = db.get(Message, user_msg_id)
    agent_msg = db.get(Message, agent_msg_id)
 
    return AskResponse(
        user_message=MessageResponse.model_validate(user_msg),
        agent_message=MessageResponse.model_validate(agent_msg),
    )

def submit_quiz(db: Session, chat_id: str, submission: QuizSubmit, user: User,
) -> QuizResultResponse:
    """
    Califica las respuestas del quiz y guarda el resultado.
 
    Compara las respuestas del usuario contra correct_answer en
    Chat.generated_content.quiz y calcula el score.
    """
    chat = get_chat(db, chat_id, user)
 
    if not chat.generated_content or not chat.generated_content.get("quiz"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Este chat no tiene quiz generado todavía.",
        )
 
    quiz_questions = chat.generated_content["quiz"]
    answers_detail = []
    correct_count = 0
  
    for answer in submission.answers:
        idx = answer.question_index
        if idx < 0 or idx >= len(quiz_questions):
            continue
 
        question = quiz_questions[idx]
        correct = question.get("correct_answer", "").upper()
        selected = answer.selected.upper()
        is_correct = selected == correct
 
        if is_correct:
            correct_count += 1
 
        answers_detail.append({
            "question_index": idx,
            "question": question.get("question", ""),
            "selected": selected,
            "correct": correct,
            "is_correct": is_correct,
            "explanation": question.get("explanation", ""),
        })
 
    result = QuizResult(
        chat_id=chat_id,
        score=correct_count,
        total_questions=len(quiz_questions),
        answers=answers_detail,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
 
    return QuizResultResponse.model_validate(result)
 
