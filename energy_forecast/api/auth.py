"""Authentication and authorization module"""

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict
import os
from dotenv import load_load

# Load environment variables
load_dotenv()

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security schemas
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthHandler:
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(
        self,
        data: Dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def decode_token(self, token: str) -> Dict:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )

class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.rate_limit = 100  # requests per minute
        self.window = 60  # seconds
        
    def is_allowed(self, api_key: str) -> bool:
        """Check if request is within rate limit"""
        now = datetime.utcnow()
        
        # Clean up old requests
        self._cleanup(now)
        
        # Get user's requests in current window
        user_requests = self.requests.get(api_key, [])
        
        # Check rate limit
        if len(user_requests) >= self.rate_limit:
            return False
        
        # Add new request
        user_requests.append(now)
        self.requests[api_key] = user_requests
        
        return True
    
    def _cleanup(self, now: datetime):
        """Remove requests outside current window"""
        window_start = now - timedelta(seconds=self.window)
        
        for api_key in list(self.requests.keys()):
            self.requests[api_key] = [
                req_time for req_time in self.requests[api_key]
                if req_time > window_start
            ]
            
            if not self.requests[api_key]:
                del self.requests[api_key]

# Authentication dependencies
auth_handler = AuthHandler()
rate_limiter = RateLimiter()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency for getting current user from token"""
    try:
        payload = auth_handler.decode_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key and check rate limit"""
    # In production, validate against database of API keys
    if not api_key.startswith("test_"):  # Simple validation for example
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Check rate limit
    if not rate_limiter.is_allowed(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return api_key
