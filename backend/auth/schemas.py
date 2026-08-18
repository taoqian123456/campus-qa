"""
认证接口的请求/响应模型（Pydantic）。
"""
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """注册请求：用户名 + 密码（至少 6 位）"""
    username: str = Field(..., min_length=1, max_length=50, description="用户名（唯一）")
    password: str = Field(..., min_length=6, max_length=128, description="密码，至少 6 位")


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
