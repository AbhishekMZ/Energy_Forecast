from datetime import timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from .security import (
    User,
    UserCreate,
    Token,
    create_access_token,
    get_current_active_user,
    get_current_verified_user,
    verify_password,
    get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from models.database import PowerSupplier, get_db

router = APIRouter()

@router.post("/register", response_model=User)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new power supplier."""
    # Check if username exists
    if db.query(PowerSupplier).filter(PowerSupplier.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    if db.query(PowerSupplier).filter(PowerSupplier.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if license number exists
    if db.query(PowerSupplier).filter(PowerSupplier.license_number == user_data.license_number).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="License number already registered"
        )

    # Create new power supplier
    db_user = PowerSupplier(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        company_name=user_data.company_name,
        supplier_type=user_data.supplier_type,
        license_number=user_data.license_number,
        capacity_mw=user_data.capacity_mw,
        location=user_data.location,
        hashed_password=get_password_hash(user_data.password),
        is_active=True,
        is_verified=False,  # Requires admin verification
        role="supplier"
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(PowerSupplier).filter(
        PowerSupplier.username == form_data.username
    ).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: PowerSupplier = Depends(get_current_active_user)):
    return current_user

@router.get("/users", response_model=List[User])
async def read_users(
    current_user: PowerSupplier = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return db.query(PowerSupplier).all()

@router.put("/verify/{username}")
async def verify_user(
    username: str,
    current_user: PowerSupplier = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Verify a power supplier (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    user = db.query(PowerSupplier).filter(PowerSupplier.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_verified = True
    db.commit()
    return {"message": f"User {username} has been verified"}
