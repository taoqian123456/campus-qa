"""
系统设置接口（仅 superadmin）：管理员注册密令的查询与设置。

密令存 bcrypt 哈希于 site_settings（key=INVITE_CODE_KEY）：
- 有哈希 = 密令注册开启；invite_code 留空 = 关闭（不校验，注册时填密令一律 400）；
- 设置后立即生效（注册接口每次现查 site_settings）。
"""
from datetime import datetime

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from auth.deps import require_superadmin
from auth.security import hash_password
from database import get_db
from models import INVITE_CODE_KEY, SiteSetting, User

router = APIRouter(prefix="/api/admin/settings", tags=["管理端-系统设置"])


class InviteCodeUpdate(BaseModel):
    """密令更新请求：留空 = 关闭密令注册"""

    invite_code: str = Field("", max_length=100, description="新管理员注册密令（留空关闭）")


@router.get("/invite")
def get_invite_setting(
    db: Session = Depends(get_db),
    su: User = Depends(require_superadmin),
):
    """查询密令是否已设置（只返回开关状态，不回显哈希）"""
    row = db.query(SiteSetting).filter(SiteSetting.key == INVITE_CODE_KEY).first()
    return {"enabled": bool(row and row.value)}


@router.put("/invite")
def update_invite_setting(
    req: InviteCodeUpdate,
    db: Session = Depends(get_db),
    su: User = Depends(require_superadmin),
):
    """设置密令（存 bcrypt 哈希）或关闭密令注册（清空 value）"""
    row = db.query(SiteSetting).filter(SiteSetting.key == INVITE_CODE_KEY).first()
    new_value = hash_password(req.invite_code) if req.invite_code else ""
    if row is None:
        row = SiteSetting(key=INVITE_CODE_KEY, value=new_value)
        db.add(row)
    else:
        row.value = new_value
        # onupdate 只对出现在 UPDATE 语句里的列生效，这里 value 是唯一脏列，需显式刷新时间
        row.updated_at = datetime.now()
    db.commit()
    return {"message": "管理员注册密令已设置" if req.invite_code else "管理员注册密令已关闭", "enabled": bool(req.invite_code)}
