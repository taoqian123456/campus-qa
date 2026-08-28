"""
认证接口的请求/响应模型（Pydantic）。
"""
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """注册请求：用户名 + 密码（至少 6 位）+ 可选管理员密令。

    invite_code 留空 -> 普通 user；填对（与 site_settings 里的哈希匹配）-> admin；
    填错或密令功能未开启 -> 400。
    """
    username: str = Field(..., min_length=1, max_length=50, description="用户名（唯一）")
    password: str = Field(..., min_length=6, max_length=128, description="密码，至少 6 位")
    invite_code: str | None = Field(None, max_length=100, description="管理员注册密令（可选，留空为普通用户）")


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class UserOut(BaseModel):
    """注册成功时返回的用户信息（不含密码）"""
    id: int
    username: str
    role: str

    model_config = {"from_attributes": True}


class RegisterResponse(BaseModel):
    """注册成功响应"""
    message: str = "注册成功"
    user: UserOut


class Token(BaseModel):
    """登录成功返回的 JWT"""
    access_token: str
    token_type: str = "bearer"
    role: str  # user / admin，前端据此显示管理后台入口
