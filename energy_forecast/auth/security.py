from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, constr, validator, Field
from models.database import SupplierType, PowerSupplier, get_db
from sqlalchemy.orm import Session

# Configuration
SECRET_KEY = "YOUR_SECRET_KEY_HERE"  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    full_name: constr(min_length=1, max_length=100)
    company_name: constr(min_length=1, max_length=100)
    supplier_type: SupplierType
    license_number: constr(min_length=5, max_length=50)
    capacity_mw: float = Field(..., gt=0)
    location: constr(min_length=1, max_length=100)

    @validator('capacity_mw')
    def validate_capacity(cls, v):
        if v <= 0:
            raise ValueError('Capacity must be greater than 0 MW')
        return v

    @validator('license_number')
    def validate_license(cls, v):
        if not v.isalnum():
            raise ValueError('License number must be alphanumeric')
        return v.upper()

class UserCreate(UserBase):
    password: constr(min_length=8)

class User(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    role: str
    created_at: datetime
    
    class Config:
        orm_mode = True

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = db.query(PowerSupplier).filter(
        PowerSupplier.username == token_data.username
    ).first()
    
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: PowerSupplier = Depends(get_current_user)
) -> PowerSupplier:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_verified_user(
    current_user: PowerSupplier = Depends(get_current_active_user)
) -> PowerSupplier:
    if not current_user.is_verified:
        raise HTTPException(
            status_code=400,
            detail="User not verified. Please wait for admin verification."
        )
    return current_user
