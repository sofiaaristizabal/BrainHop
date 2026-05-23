"""
base.py contiene el database engine, el session factory y el base model classs
 - Crea el engine de SQLAlchemy (la conección a postgres)
 - Crea el sessionLocal factory (para abrir y cerrar sesiones por request a la base de datos)
 - Define la declarative Base de las cuales todos los modelos ORM heredan
"""

from sqlalchemy import Column, String, create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import config

#Maneja la conección a postgres, se corre una vez cuando se importa el modulo
engine = create_engine(
    config.DATABASE_URL,
    poolclass=NullPool, # NullPool is used here to avoid issues with async/multiprocess environments
    echo=True # Echo=True logs every SQL statement to the console — very useful during development
)

#Session factory: la session local es una clae. Cada vez que se llama se crea una nueva sesión a la base de datos 
SessionLocal = sessionmaker(
    bind = engine,
    autocommit = False, # autocommit=False means changes are NOT saved until you explicitly call session.commit(). This is what you want: you control when data is persisted.
    autoflush=False # autoflush=False means SQLAlchemy won't automatically sync pending changes to the DB before queries. We manage this manually for clarity.
)

#Declaramos la clase Base de la cual van a heredar todos los modelos, SQLAlchemy lo usa para mantener registro de que clases represnetan tablas en la base de datos, como si fuera un @Entity()
class Base(DeclarativeBase):
    pass

#Creamos un metodo getdb que es una función de dependencia que va autilizar fastapi 
def get_db():
    db = SessionLocal() #Route handlers declare it with Depends(get_db) and receive a live session. The try/finally block ensures the session is always closed after the request, even if an exception occurs.
    try:
        yield db
    finally:
        db.close()

#Creamos un metodo para verificar que la conexión este viva
def check_db_connection() ->bool:
    #Corremos una query cualquiera para ver que la conexión este funcionando
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True