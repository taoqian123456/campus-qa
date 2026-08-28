"""
高校学生事务智能问答系统 - FastAPI 入口
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from admin.documents import router as admin_router
from admin.kb_health import router as kb_health_router
from admin.settings import router as admin_settings_router
from admin.stats import router as admin_stats_router
from admin.users import router as admin_users_router
from auth.deps import get_current_user
from auth.routes import router as auth_router
from chat.routes import router as chat_router
from config import APP_NAME, APP_VERSION
from init_db import migrate_add_columns, seed_invite_code
from models import User

# 前端单文件（backend 同级的 frontend/index.html），由后端同源提供
FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动钩子：建表、轻量迁移、密令哈希入库（幂等，见 init_db.py）"""
    from database import init_db

    init_db()
    migrate_add_columns()
    seed_invite_code()
    yield


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)

# 注册路由：认证 / 管理端（知识库文档、用户、统计）/ 会话
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(admin_users_router)
app.include_router(admin_stats_router)
app.include_router(admin_settings_router)
app.include_router(kb_health_router)
app.include_router(chat_router)

# 允许跨域（前端单文件直接打开时也需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """健康检查：验证服务是否启动"""
    return {"status": "ok", "service": "campus-qa", "version": APP_VERSION}


@app.get("/")
def index():
    """提供前端页面：同源访问（http://127.0.0.1:8000/），
    避免浏览器用 file:// 打开时 localStorage 被拦截、页面无法挂载的问题。"""
    if not FRONTEND_INDEX.exists():
        return {"message": "前端文件不存在：请确认 frontend/index.html 就位", "path": str(FRONTEND_INDEX)}
    return FileResponse(FRONTEND_INDEX)


@app.get("/api/auth/me", tags=["认证"])
def me(current_user: User = Depends(get_current_user)):
    """测试接口：验证 get_current_user 依赖，返回当前登录用户。"""
    return {"id": current_user.id, "username": current_user.username, "role": current_user.role}
