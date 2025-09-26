"""
JWT Token Handler - Production Implementation
Verantwoordelijk voor JWT creatie en verificatie zoals gespecificeerd in FASE 1 STAP 1.2
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt, JWTError
from passlib.context import CryptContext
from core.config.unified_config import config
import secrets
import logging

logger = logging.getLogger(__name__)

class JWTHandler:
    """
    Production-grade JWT handler voor authenticatie
    Integrated with unified configuration system
    """

    def __init__(self):
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.algorithm = "HS256"
        self.secret_key = self._get_secret_key()
        self.access_token_expire_minutes = self.config.jwt_expire_minutes

    def _get_secret_key(self) -> str:
        """Get JWT secret key with validation"""
        secret = self.config.jwt_secret_key

        if not secret:
            if self.config.is_production():
                raise ValueError("JWT_SECRET_KEY must be set in production")
            # Generate random secret for development
            secret = secrets.token_urlsafe(32)
            logger.warning(f"Using generated secret for development: {secret[:8]}...")

        # Validate secret strength
        if self.config.is_production():
            if len(secret) < 32:
                raise ValueError("JWT secret must be at least 32 characters in production")
            if secret.startswith("dev-") or "change-in-production" in secret:
                raise ValueError("Development secret detected in production environment")

        return secret

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Creëer een JWT access token

        Args:
            data: Data om in de token te encoderen (bijv. {"sub": "user@email.com"})
            expires_delta: Optionele custom expiry tijd

        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({"exp": expire})

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verificeer en decodeer een JWT token

        Args:
            token: JWT token string om te verificeren

        Returns:
            Decoded payload als token geldig is, anders None
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None

    def hash_password(self, password: str) -> str:
        """
        Hash een wachtwoord met bcrypt

        Args:
            password: Plain text wachtwoord

        Returns:
            Gehashed wachtwoord
        """
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verificeer een wachtwoord tegen de hash

        Args:
            plain_password: Plain text wachtwoord
            hashed_password: Gehashed wachtwoord uit database

        Returns:
            True als wachtwoord correct is, anders False
        """
        return self.pwd_context.verify(plain_password, hashed_password)


# Global instance
jwt_handler = JWTHandler()