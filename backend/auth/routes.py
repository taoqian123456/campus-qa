"""
认证路由：POST /api/auth/register、POST /api/auth/login。
"""
import time
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth import schemas, security
from database import get_db
from models import SiteSetting, User

router = APIRouter(prefix="/api/auth", tags=["认证"])

INVITE_CODE_KEY = "invite_code_hash"

# ---------- 密令爆破防护（内存级，可选加固） ----------
# 同一 IP 连续失败 5 次锁定 10 分钟；进程重启即清零。
# 内存字典对单实例部署（本项目形态）足够，多实例部署应换 Redis。
INVITE_FAIL_LIMIT = 5
INVITE_LOCK_SECONDS = 600
_invite_failures: dict[str, list[float]] = {}  # ip -> 失败时间戳列表
_invite_fail_lock = Lock()


def client_ip(request: Request) -> str:
    """取客户端 IP：优先 X-Forwarded-For 链的首个地址（反代/Compose 场景），
    否则用直连 socket 地址；只信任一层代理注入。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_invite_limit(ip: str) -> None:
    """密令校验前的限流检查：锁定期内直接 429；清理过期记录由每次调用顺带完成。"""
    with _invite_fail_lock:
        now = time.time()
        failures = [t for t in _invite_failures.get(ip, []) if now - t < INVITE_LOCK_SECONDS]
        _invite_failures[ip] = failures
        if len(failures) >= INVITE_FAIL_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="密令尝试次数过多，请 10 分钟后再试",
            )


def record_invite_failure(ip: str) -> None:
    """记录一次密令校验失败（仅当密令非空且确实填错时调用）。"""
    with _invite_fail_lock:
        _invite_failures.setdefault(ip, []).append(time.time())


def resolve_role(invite_code: str | None, request: Request, db: Session) -> str:
    """按密令确定注册角色：留空 -> user；与 site_settings 哈希匹配 -> admin；其余情况抛 400。

    校验失败会累计该 IP 的失败次数（防爆破，见 check_invite_limit）。
    """
    if not invite_code:
        return "user"
    ip = client_ip(request)
    check_invite_limit(ip)
    stored = db.scalar(select(SiteSetting).where(SiteSetting.key == INVITE_CODE_KEY))
    if stored is None or not stored.value:
        raise HTTPException(status_code=400, detail="管理员密令功能未开启")
    if not security.verify_password(invite_code, stored.value):
        record_invite_failure(ip)
        raise HTTPException(status_code=400, detail="管理员密令错误")
    return "admin"


@router.post("/register", response_model=schemas.RegisterResponse, status_code=201)
def register(req: schemas.RegisterRequest, request: Request, db: Session = Depends(get_db)):
    """注册：用户名重复返回 400；密令正确 -> admin，不填 -> user。"""
    exists = db.scalar(select(User).where(User.username == req.username))
    if exists:
        raise HTTPException(status_code=400, detail="用户名已存在")

    user = User(
        username=req.username,
        hashed_password=security.hash_password(req.password),
        role=resolve_role(req.invite_code, request, db),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return schemas.RegisterResponse(user=user)


@router.post("/login", response_model=schemas.Token)
def login(req: schemas.LoginRequest, db: Session = Depends(get_db)):
    """登录：校验用户名密码，成功返回 JWT（7 天有效）；被禁用的账号返回 403。"""
    user = db.scalar(select(User).where(User.username == req.username))
    if not user or not security.verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已被禁用，请联系管理员")
    return schemas.Token(access_token=security.create_access_token(user.id), role=user.role)
