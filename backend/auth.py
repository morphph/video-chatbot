import os
from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import json
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBearer()

class ClerkAuth:
    def __init__(self):
        self.secret_key = os.getenv("CLERK_SECRET_KEY")
        self.publishable_key = os.getenv("CLERK_PUBLISHABLE_KEY")
        if not self.secret_key:
            logger.warning("CLERK_SECRET_KEY not set - authentication will be disabled")
    
    @lru_cache(maxsize=100)
    async def verify_token(self, token: str) -> Optional[dict]:
        """Verify Clerk JWT token"""
        if not self.secret_key:
            # Return a mock user for development
            return {"sub": "temp-user-id", "email": "dev@example.com"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.clerk.dev/v1/sessions/verify",
                    headers={
                        "Authorization": f"Bearer {self.secret_key}",
                        "Content-Type": "application/json",
                    },
                    params={"token": token}
                )
                
                if response.status_code == 200:
                    session_data = response.json()
                    return session_data.get("user")
                else:
                    logger.error(f"Token verification failed: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
        """Get current user from token"""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        user = await self.verify_token(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return user

# Global auth instance
clerk_auth = ClerkAuth()

# Dependency for protected routes
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    return await clerk_auth.get_current_user(credentials)

# Optional auth dependency (allows unauthenticated access for development)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[dict]:
    """Optional authentication - returns None if not authenticated"""
    if not credentials:
        return {"sub": "anonymous", "email": "anonymous@example.com"}
    
    try:
        return await clerk_auth.get_current_user(credentials)
    except HTTPException:
        return {"sub": "anonymous", "email": "anonymous@example.com"}