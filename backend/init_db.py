"""
初始化数据库：运行 python init_db.py 即可建表。
验证：数据库目录下出现 campus_qa.db 文件，或安装 DB Browser for SQLite 查看。

对已有库做轻量迁移（SQLite 简单方案）：
- 新表用 SQLAlchemy create_all 直接建；
- messages 表新增列（reply_feedback / reply_reason / provider）用 PRAGMA table_info 检查后
  ALTER TABLE ADD COLUMN 补齐，无需重建表、不丢数据。
"""
from sqlalchemy import inspect, text

from database import engine, init_db


def migrate_add_columns():
    """给已有表补新列：只对"模型里有、表里没有"的列执行 ALTER TABLE ADD COLUMN。"""
    from models import Document, Message  # noqa: F401  确保模型注册

    insp = inspect(engine)
    additions = {
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


if __name__ == "__main__":
    init_db()
    migrate_add_columns()
    print("[OK] 数据库初始化完成")
