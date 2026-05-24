"""
main.py es el punto de entrada de la aplicación 
Registra todos los routers, condigura CORS y verifica las conexiones a postgres y Ollama 
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import config
from app.db.base import engine, check_db_connection
from app.db.base import Base
from app.db import models
from app.db.vector_store import EmbeddingChunk
from app.api.controllers import chats, documents, messages, users

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Este codigo corre al arrancar la app y al apagarla"""

    logger.info("Iniciando BrainHop backend")

    #Verificamos la conección a postgres
    try:
        check_db_connection()
        logger.info("Conexión a PostgreSQL: OK")
    except Exception as e:
        logger.error(f"No se pudo conectar a PostgreSQL: {e}")
        raise

    #Creamos las tablas si no existen
    Base.metadata.create_all(bind=engine)
    logger.info("Tablas verificadas/creadas.")

    logger.info("BrainHop backend listo.")
    yield
    # --- Shutdown ---
    logger.info("Apagando BrainHop backend.")

app = FastAPI(
    title="BrainHop API",
    description="Backend para BrainHop — agente educativo para niños con ADHD",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allowed_credentials=True,
    allowed_methods=["*"],
    allow_headers=["*"],
)

#Registramos los routers
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(chats.router,     prefix="/api/chats",     tags=["chats"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(messages.router,  prefix="/api/messages",  tags=["messages"])

@app.get("/health")
def health_check():
    """Endpoint para que el orquestador (Docker) verifique que la app está viva."""
    return {"status": "ok", "service": "brainhop-backend"}