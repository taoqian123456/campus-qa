"""
用户管理接口（admin 及以上）：列表（含角色统计）、删除。
超管专属（superadmin）：改角色、启用/禁用账号。

权限分层：
- admin 只能删普通 user；
- superadmin 可删 admin，但任何角色都不能删 superadmin（含自己）；
- 改角色/状态只有 superadmin 能操作，且不能动 superadmin（防锁死系统）。
"""
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth.deps import require_admin, require_superadmin
from database import get_db
from models import ChatSession, Message, User

router = APIRouter(prefix="/api/admin/users", tags=["管理端-用户"])

SUPERADMIN = "superadmin"


class RoleUpdate(BaseModel):
    """改角色请求：只能 user <-> admin，superadmin 不在允许范围内"""

    role: str = Field(..., pattern="^(user|admin)$", description="目标角色：user 或 admin")


class StatusUpdate(BaseModel):
    """启用/禁用请求"""

    is_active: bool


def _user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "role": u.role, "is_active": u.is_active, "created_at": u.created_at}


@router.get("")
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """列出全部用户（按注册时间倒序）+ 角色统计。"""
    users = db.scalars(select(User).order_by(User.created_at.desc(), User.id.desc())).all()
    stats = {
        "user_count": sum(1 for u in users if u.role == "user"),
        "admin_count": sum(1 for u in users if u.role == "admin"),
        "superadmin_count": sum(1 for u in users if u.role == SUPERADMIN),
    }
    return {"users": [_user_out(u) for u in users], "stats": stats}


@router.patch("/{user_id}/role")
def update_role(
    user_id: int,
    req: RoleUpdate,
    db: Session = Depends(get_db),
    su: User = Depends(require_superadmin),
):
    """改角色（仅 superadmin）：不能改 superadmin（含自己降权），改完返回新角色。"""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.role == SUPERADMIN:
        raise HTTPException(status_code=403, detail="不能修改超级管理员的角色")
    if user.role == req.role:
        raise HTTPException(status_code=400, detail=f"该用户当前已是 {req.role}")
    user.role = req.role
    db.commit()
    return {"message": "角色已更新", "id": user.id, "username": user.username, "role": user.role}


@router.patch("/{user_id}/status")
def update_status(
    user_id: int,
    req: StatusUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """启用/禁用账号（admin 及以上）：不能禁用 superadmin；禁用后该用户登录时被拒。"""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.role == SUPERADMIN and not req.is_active:
        raise HTTPException(status_code=403, detail="不能禁用超级管理员")
    if user.is_active == req.is_active:
        raise HTTPException(status_code=400, detail=f"该账号当前已{'启用' if req.is_active else '禁用'}")
    user.is_active = req.is_active
    db.commit()
    return {"message": "账号已启用" if req.is_active else "账号已禁用",
            "id": user.id, "username": user.username, "is_active": user.is_active}


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """删除用户：同时删除其全部会话与消息。

    权限分层：不能删自己；admin 只能删 user；superadmin 可删 admin；
    任何角色都不能删 superadmin。
    """
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="不能删除自己")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.role == SUPERADMIN:
        raise HTTPException(status_code=403, detail="不能删除超级管理员")
    if user.role == "admin" and admin.role != SUPERADMIN:
        raise HTTPException(status_code=403, detail="仅超级管理员可删除管理员")

    # 删会话（其下消息随 sessions 一并删除）
    session_ids = db.scalars(select(ChatSession.id).where(ChatSession.user_id == user_id)).all()
    if session_ids:
        db.query(Message).filter(Message.session_id.in_(session_ids)).delete()
        db.query(ChatSession).filter(ChatSession.user_id == user_id).delete()

    db.delete(user)
    db.commit()
    return {"message": "用户已删除", "id": user_id, "username": user.username}
