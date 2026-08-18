"""
创建管理员账号（可单独运行）。

用法：venv\\Scripts\\python.exe create_admin.py
交互式输入用户名和密码（密码不回显），role='admin'。
用户名已存在时报错退出，不覆盖已有账号。
"""
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from database import SessionLocal
from models import User
from auth.security import hash_password

MIN_PASSWORD_LEN = 6


def main():
    username = input("管理员用户名: ").strip()
    if not username:
        print("❌ 用户名不能为空")
        sys.exit(1)

    db = SessionLocal()
    try:
        exists = db.query(User).filter(User.username == username).first()
        if exists:
            print(f"❌ 用户名 {username} 已存在（role={exists.role}），请换一个名字")
            sys.exit(1)

        password = getpass.getpass("密码（至少 6 位，输入不回显）: ")
        if len(password) < MIN_PASSWORD_LEN:
            print(f"❌ 密码至少 {MIN_PASSWORD_LEN} 位")
            sys.exit(1)
        if getpass.getpass("再输一次确认: ") != password:
            print("❌ 两次输入的密码不一致")
            sys.exit(1)

        user = User(username=username, hashed_password=hash_password(password), role="admin")
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"✅ 管理员 {username} 创建成功（id={user.id}, role={user.role}）")
    finally:
        db.close()


if __name__ == "__main__":
    main()
