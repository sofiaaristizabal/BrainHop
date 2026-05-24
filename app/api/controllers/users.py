"""
users.py es el endpoint del usuario 
Solo ponemos un GET, no necesitamos un endpoint de post para crear usuarios ya que 
cuando el GET llama al metodo get_current_user que creamos en depends.py, cuando el
usuario no existe, el metodo automaticamente lo crea y lo agrega a la base de datos
"""

from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.db.models import User
from app.api.schemas.schemas import UserResponse

router = APIRouter()

@router.get("/user", response_model= UserResponse)
def get_user(current_user: User = Depends(get_current_user)): #Depends tells FastAPI to execute that function before running your endpoint code. get_current_user, notices it needs things like Depends(bearer_scheme) and Depends(get_db), and recursively solves all those needs for you. fastAPI extracts the token from the browser request headers automatically.
    """Retorna el perfil del usuario autenticado"""
    return current_user