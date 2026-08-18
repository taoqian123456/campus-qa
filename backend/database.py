"""
数据库连接：引擎、会话、初始化
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

# check_same_thread=False：SQLite 在 FastAPI 异步场景下必需
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db():
    """建表（幂等：表已存在则跳过）。"""
    import models  # noqa: F401  确保所有表模型被注册
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI 依赖：每个请求一个数据库会话，用完自动关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
