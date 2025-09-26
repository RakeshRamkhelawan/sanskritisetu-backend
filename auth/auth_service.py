"""
Authentication Service - Production Implementation
Database operaties voor user authenticatie zoals gespecificeerd in FASE 1 STAP 1.2
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from core.database.models import User, Role
from core.auth.jwt_handler import jwt_handler
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """User creation model"""
    email: EmailStr
    password: str
    role_name: str = "user"


class UserResponse(BaseModel):
    """User response model"""
    id: int
    email: str
    role_id: int

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "bearer"


class AuthService:
    """
    Production-grade authentication service
    """

    async def create_user(self, db: AsyncSession, user_data: UserCreate) -> Optional[UserResponse]:
        """
        Creëer een nieuwe gebruiker

        Args:
            db: Database sessie
            user_data: User data om aan te maken

        Returns:
            UserResponse als succesvol, anders None
        """
        # Check if user already exists
        result = await db.execute(select(User).where(User.email == user_data.email))
        if result.scalar_one_or_none():
            return None  # User already exists

        # Get role
        role_result = await db.execute(select(Role).where(Role.name == user_data.role_name))
        role = role_result.scalar_one_or_none()
        if not role:
            # Create default user role if it doesn't exist
            role = Role(name=user_data.role_name)
            db.add(role)
            await db.flush()

        # Create user
        hashed_password = jwt_handler.hash_password(user_data.password)
        user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            role_id=role.id
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        return UserResponse(id=user.id, email=user.email, role_id=user.role_id)

    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> Optional[UserResponse]:
        """
        Authenticeer een gebruiker met email en wachtwoord

        Args:
            db: Database sessie
            email: User email
            password: Plain text wachtwoord

        Returns:
            UserResponse als authenticatie succesvol, anders None
        """
        # Get user from database
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user:
            return None

        # Verify password
        if not jwt_handler.verify_password(password, user.hashed_password):
            return None

        return UserResponse(id=user.id, email=user.email, role_id=user.role_id)

    async def get_user_by_email(self, db: AsyncSession, email: str) -> Optional[UserResponse]:
        """
        Haal gebruiker op uit database op basis van email

        Args:
            db: Database sessie
            email: User email

        Returns:
            UserResponse als gevonden, anders None
        """
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user:
            return None

        return UserResponse(id=user.id, email=user.email, role_id=user.role_id)

    def create_access_token(self, user: UserResponse) -> TokenResponse:
        """
        Creëer access token voor gebruiker

        Args:
            user: User data

        Returns:
            TokenResponse met access token
        """
        access_token = jwt_handler.create_access_token(
            data={"sub": user.email, "user_id": user.id, "role_id": user.role_id}
        )
        return TokenResponse(access_token=access_token)


# Global instance
auth_service = AuthService()