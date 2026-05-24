"""
deps.py es el archivo donde guardamos las dependencias compartidas entre los controladores, o como lo llaman en python, los routers
Tenemos las dependencias que FastAPI inyecta a los route handlers:
- get_db() sesión de base de datos por request
- get_current_user() usuario autenticado vía token de Keycloak
"""

import logging
from functools import lru_cache
import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.config import config
from app.db.base import get_db
from app.db.models import User

logger = logging.getLogger(__name__)
 
# HTTPBearer extrae el token del header "Authorization: Bearer <token>"
bearer_scheme = HTTPBearer()

#Keycloak JWKS, lo usamos para verificar tokens JWT
@lru_cache(maxsize=1)
def _get_keycloak_public_keys() -> dict:
    """
    Descargamos las claves publicas JWKS de keycloak para verificar tokens. Cacheado con lru_cache para no hacer una petición HTTP en cada request.
    """

    jwks_url = (
        f"{config.KEYCLOAK_URL}/realms/{config.KEYCLOAK_REALM}"
        f"/protocol/openid-connect/certs"
    )
    try:
        response = httpx.get(jwks_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"No se pudieron obtener las claves JWKS de Keycloak: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio de autenticación no disponible.",
        )
    
def _decode_token(token:str) -> dict:
    """
    Verifica y decodifica el JWT de keycloak, valida la firma, expiracion y audience
    """

    try:
        jwks = _get_keycloak_public_keys()
        payload = jwt.decode(  # jose.jwt.decode busca automáticamente la clave correcta en el JWKS usando el "kid" (key ID) del header del token
            token,
            jwks,
            algorithms=["RS256"],
            audience=config.KEYCLOAK_CLIENT_ID,
            options={"verify_at_hash": False},
        )
        return payload
    except JWTError as e:
        logger.warning(f"Token jwt invalido: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or not valid",
            headers={"WWW-Authenticate": "Bearer"}
        )

#Dependencia principal de autenticación
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme), db:Session = Depends(get_db)) -> User:
    """
    Extrae y valida el token JWT y luego retorna el User de la DB
    Si el usuario no existe en nuestra DB todavía (primer login),
    lo crea automáticamente usando los claims del token.
    Esto elimina la necesidad de un endpoint separado de "registro"
    """

    token = credentials.credentials
    payload = _decode_token(token)
 
    keycloak_id: str = payload.get("sub")
    email: str = payload.get("email", "")
    username: str = payload.get("preferred_username", email)
 
    if not keycloak_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token sin identificador de usuario.",
        )
 
    # Buscar usuario existente
    user = db.query(User).filter(User.keycloak_id == keycloak_id).first()
 
    if user is None:
        # Primer login: crear el usuario automáticamente
        logger.info(f"Primer login para keycloak_id={keycloak_id}, creando usuario.")
        user = User(
            keycloak_id=keycloak_id,
            email=email,
            username=username,
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
 
    elif not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cuenta de usuario desactivada.",
        )
 
    return user

__all__ = ["get_db", "get_current_user"]