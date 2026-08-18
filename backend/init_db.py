"""
初始化数据库：运行 python init_db.py 即可建表。
验证：数据库目录下出现 campus_qa.db 文件，或安装 DB Browser for SQLite 查看。
"""
from database import init_db

if __name__ == "__main__":
    init_db()
    print("✅ 数据库初始化完成")
