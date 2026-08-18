"""
认证路由：POST /api/auth/register、POST /api/auth/login。
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import schemas, security
from database import get_db
from models import User

router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post("/register", response_model=schemas.RegisterResponse, status_code=201)
def register(req: schemas.RegisterRequest, db: Session = Depends(get_db)):
    """注册：用户名重复返回 400。"""
    exists = db.scalar(select(User).where(User.username == req.username))
    if exists:
        raise HTTPException(status_code=400, detail="用户名已存在")

    user = User(username=req.username, hashed_password=security.hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return schemas.RegisterResponse(user=user)


@router.post("/login", response_model=schemas.Token)
def login(req: schemas.LoginRequest, db: Session = Depends(get_db)):
    """登录：校验用户名密码，成功返回 JWT（7 天有效）。"""
    user = db.scalar(select(User).where(User.username == req.username))
    if not user or not security.verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return schemas.Token(access_token=security.create_access_token(user.id), role=user.role)
