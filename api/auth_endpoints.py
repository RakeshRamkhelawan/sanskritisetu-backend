"""
Authentication Endpoints - Production Implementation
API endpoints voor authenticatie zoals gespecificeerd in FASE 1 STAP 1.2
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from core.database.database import get_db
from core.auth import auth_service, UserCreate, UserResponse, TokenResponse, jwt_handler
from typing import Optional

router = APIRouter(prefix="/auth", tags=["authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
security = HTTPBearer()


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> UserResponse:
    """
    Dependency om huidige gebruiker op te halen uit JWT token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = jwt_handler.verify_token(token)
    if payload is None:
        raise credentials_exception

    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception

    user = await auth_service.get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception

    return user


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info - Compatible with main.py expectations"""
    token = credentials.credentials
    payload = jwt_handler.verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def verify_admin(current_user: dict = Depends(verify_token)):
    """Verify admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Registreer een nieuwe gebruiker
    """
    user = await auth_service.create_user(db, user_data)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    return user


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Login endpoint - valideer email/wachtwoord en return access token
    """
    user = await auth_service.authenticate_user(db, form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return auth_service.create_access_token(user)


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_user)):
    """
    Test endpoint - haal huidige gebruiker info op
    """
    return current_user


@router.get("/protected-test")
async def protected_test_endpoint(current_user: UserResponse = Depends(get_current_user)):
    """
    Test endpoint voor authenticatie verificatie
    """
    return {"message": f"Hello {current_user.email}, you are authenticated!", "user_id": current_user.id}