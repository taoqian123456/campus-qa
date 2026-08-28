"""
数据表模型（5 张核心表）：
users 用户 / sessions 会话 / messages 消息 / documents 知识库文档 / site_settings 系统设置
注意：会话类名用 ChatSession（避免和 SQLAlchemy 的 Session 混淆）。
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    # 三级角色：user 普通用户 / admin 管理员（密令注册）/ superadmin 超级管理员（create_admin.py 创建）
    role: Mapped[str] = mapped_column(String(20), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # 预留：禁用用户（False 时拒绝登录）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ChatSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(100), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 本条消息使用的模型厂商（config.LLM_PROVIDERS 的 key，如 deepseek/zhipu）；user 消息无意义，
    # assistant 消息用于前端展示"本条回答由哪个模型生成"；旧数据默认 deepseek
    provider: Mapped[str] = mapped_column(String(30), default="deepseek")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    # 回答反馈：up 点赞 / down 点踩（仅 assistant 消息有效）；reply_reason 是点踩原因
    reply_feedback: Mapped[str | None] = mapped_column(String(10), nullable=True)
    reply_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending / indexed / failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    # 知识库体检：分块数（索引重建时同步）与检索命中次数（每次问答 top-k 命中时累加）
    chunk_count: Mapped[int] = mapped_column(default=0)
    hit_count: Mapped[int] = mapped_column(default=0)


class SiteSetting(Base):
    """系统设置键值表：如 invite_code_hash（管理员注册密令的 bcrypt 哈希）"""

    __tablename__ = "site_settings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


# 管理员注册密令在 site_settings 里的 key（init_db / auth / admin.settings 共用）
INVITE_CODE_KEY = "invite_code_hash"
