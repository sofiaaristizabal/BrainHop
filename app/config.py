"""
config.py sentraliza la configuración de la aplicación:
    - Todos los archivos que necesiten cualquier variable de entorno deben taerlas de aqui (ayuda a debugging)
    - Se utiliza pydantic para validar el tipo de datos que se traen (en caso de que se cambie algo en el env validar su tipo para que no rompa la app)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class Config(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    COLLECTION_NAME: str = "documentos"
    CHAT_MODEL: str = "llama3.1:8b"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_URL: str = os.getenv("OLLAMA_URL") 
    EMBEDDING_DIM: int = 768 #The number 768 became a default industry standard due to Google's BERT architecture, which revolutionized natural language processing.
    KEYCLOAK_URL: str = os.getenv("KEYCLOAK_URL")
    KEYCLOAK_REALM: str = os.getenv("KEYCLOAK_REALM")
    KEYCLOAK_CLIENT_ID: str = os.getenv("KEYCLOAK_CLIENT_ID")
    KEYCLOAK_CLIENT_SECRET: str = os.getenv("KEYCLOAK_CLIENT_SECRET")

    CHUNK_SIZE:int = 1000 #1000 chars ≈ ~250 tokens
    CHUNK_OVERLAP:int=150 #para evitar cortar información importante y contexto

    #Estrategia para evitar alucinaciones 
    MIN_SIMILARITY_THRESHOLD: float = 0.4 #Minimum cosine similarity (0.0–1.0) for a chunk to be considered relevant. Chunks below this score are silently dropped — the agent won't use them.

    #Numero de chunks que se retorna por search query
    TOP_K_RESULTS: int = 5

    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024 #Maximo tamaño permitido para los documentos

    # Tipos de documentos permitidos (Multipurpose Internet Mail Extensions. t is an ancient internet standard used everywhere today (especially in HTTP requests and web browsers). It is a standardized way for computers to label the nature and format of a file so they know how to handle it safely before opening it.)
    ALLOWED_MIME_TYPES: list[str] = [
        "application/pdf", #A file extension like .pdf or .png is just a couple of characters at the end of a filename. Anyone can rename a malicious executable file from virus.exe to homework.pdf. If your backend only checked the extension, it would easily be tricked.MIME types use a strict [type]/[subtype] structure
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", #This is the official, standardized MIME type for a modern Microsoft Word document (.docx).
    ]

    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS").split(',')
    
config = Config()