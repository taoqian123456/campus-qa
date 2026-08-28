"""
初始化数据库：运行 python init_db.py 即可建表。
验证：数据库目录下出现 campus_qa.db 文件，或安装 DB Browser for SQLite 查看。

对已有库做轻量迁移（SQLite 简单方案）：
- 新表用 SQLAlchemy create_all 直接建；
- users/messages/documents 表新增列用 PRAGMA table_info 检查后
  ALTER TABLE ADD COLUMN 补齐，无需重建表、不丢数据。

管理员注册密令：首次运行且 site_settings 里还没有 invite_code_hash 时，
把 config.ADMIN_INVITE_CODE（非空）bcrypt 哈希后写入 site_settings；
之后密令以库里为准，改 .env 不再生效。
"""
from sqlalchemy import inspect, text

from database import engine, init_db

INVITE_CODE_KEY = "invite_code_hash"


def migrate_add_columns():
    """给已有表补新列：只对"模型里有、表里没有"的列执行 ALTER TABLE ADD COLUMN。"""
    from models import Document, Message, User  # noqa: F401  确保模型注册

    insp = inspect(engine)
    additions = {
        "users": {
            "is_active": "BOOLEAN NOT NULL DEFAULT 1",
        },
        "messages": {
            "reply_feedback": "VARCHAR(10)",
            "reply_reason": "VARCHAR(200)",
            # 模型厂商列：旧库补列时带上与模型一致的默认值（老消息统一归到 deepseek）
            "provider": "VARCHAR(30) NOT NULL DEFAULT 'deepseek'",
        },
        "documents": {
            "chunk_count": "INTEGER NOT NULL DEFAULT 0",
            "hit_count": "INTEGER NOT NULL DEFAULT 0",
        },
    }
    tables = insp.get_table_names()
    with engine.begin() as conn:
        for table, cols in additions.items():
            if table not in tables:
                continue  # 新库：create_all 会直接建全量表，无需补列
            existing = {c["name"] for c in insp.get_columns(table)}
            for col, col_type in cols.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                    print(f"迁移：{table} 表新增列 {col}")


def seed_invite_code():
    """管理员密令入库：site_settings 里没有 invite_code_hash 且配置了 ADMIN_INVITE_CODE 时，
    把密令的 bcrypt 哈希写入（只写一次，之后以库里为准）。

    ADMIN_INVITE_CODE 在函数内现取（而非模块导入时拷贝），保证配置变更后重启能读到新值。
    """
    from config import ADMIN_INVITE_CODE
    from auth.security import hash_password
    from database import SessionLocal
    from models import SiteSetting

    if not ADMIN_INVITE_CODE:
        return
    db = SessionLocal()
    try:
        exists = db.query(SiteSetting).filter(SiteSetting.key == INVITE_CODE_KEY).first()
        if exists:
            return
        db.add(SiteSetting(key=INVITE_CODE_KEY, value=hash_password(ADMIN_INVITE_CODE)))
        db.commit()
        print("密令：ADMIN_INVITE_CODE 哈希已写入 site_settings")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    migrate_add_columns()
    seed_invite_code()
    print("[OK] 数据库初始化完成")
