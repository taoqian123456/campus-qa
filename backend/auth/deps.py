"""
认证依赖：get_current_user —— 从 Authorization: Bearer 头解析 JWT 并取出当前用户；
require_admin / require_superadmin —— 角色权限控制。
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from auth.security import decode_access_token
from database import get_db
from models import User

# auto_error=False：未带 Authorization 头时由依赖统一返回 401，而不是框架默认 403
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """解析 Bearer Token -> 当前登录用户；token 无效或用户不存在返回 401。"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录：缺少 Authorization 头",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效或凭证无效",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """管理端依赖：admin 与 superadmin 都通过，其余返回 403。"""
    if current_user.role not in ("admin", "superadmin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限：仅管理员可操作",
        )
    return current_user


def require_superadmin(current_user: User = Depends(get_current_user)) -> User:
    """超级管理员专属依赖：仅 superadmin 通过，admin 返回 403。"""
    if current_user.role != "superadmin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限：仅超级管理员可操作",
        )
    return current_user
