"""
用户管理接口（仅 admin）：列表、删除。

权限说明：admin 不可删除 admin（含自己），防止把唯一的后台入口误删掉。
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from admin.documents import require_admin
from database import get_db
from models import ChatSession, Message, User

router = APIRouter(prefix="/api/admin/users", tags=["管理端-用户"])


def _user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "role": u.role, "created_at": u.created_at}


@router.get("")
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """列出全部用户（按注册时间倒序）。"""
    users = db.scalars(select(User).order_by(User.created_at.desc(), User.id.desc())).all()
    return [_user_out(u) for u in users]


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """删除用户：同时删除其全部会话与消息；admin 不可删除 admin。"""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="不能删除自己")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.role == "admin":
        raise HTTPException(status_code=400, detail="不能删除管理员账号")

    # 删会话（其下消息随 sessions 一并删除）
    session_ids = db.scalars(select(ChatSession.id).where(ChatSession.user_id == user_id)).all()
    if session_ids:
        db.query(Message).filter(Message.session_id.in_(session_ids)).delete()
        db.query(ChatSession).filter(ChatSession.user_id == user_id).delete()

    db.delete(user)
    db.commit()
    return {"message": "用户已删除", "id": user_id, "username": user.username}
