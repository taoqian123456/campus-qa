"""
三级角色 + 管理员密令 + 超管专属接口 验收测试（可单独运行）。

用法：venv\\Scripts\\python.exe test_role_system.py
在临时 SQLite 库上完整模拟验收流程：

第一轮（角色体系 + 密令初始化）：
① 无密令配置时注册（不填密令）→ user；填密令 → 400 未开启
② 配置 ADMIN_INVITE_CODE=test123 后重新初始化 → 填对 → admin，填错 → 400
③ create_admin.py 创建的超级管理员 → role=superadmin（按其写入逻辑模拟）

第二轮（超管专属接口）：
① superadmin 设置密令（PUT /api/admin/settings/invite）→ 注册新用户填密令成功为 admin
② admin 调 PUT /settings/invite 返回 403
③ 改角色 user→admin 生效（PATCH /api/admin/users/{id}/role）
④ 禁用用户后该用户登录被拒（PATCH status + login 403）
⑤ 超管不能改/删其他超管

注意：脚本会清空临时库、不碰正式 campus_qa.db。
"""
import importlib
import os
import sys
from pathlib import Path

# 控制台可能是 GBK，统一 UTF-8 输出避免 emoji 报错
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

BACKEND_DIR = Path(__file__).resolve().parent
TEST_DB = BACKEND_DIR / "role_test_tmp.db"
TEST_DB.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"
os.environ.pop("ADMIN_INVITE_CODE", None)  # 环境①：未配置密令

from fastapi.testclient import TestClient

import main  # 环境①的 app：无 ADMIN_INVITE_CODE
from auth.security import hash_password
from database import SessionLocal, engine
from init_db import seed_invite_code
from models import User

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    results.append((name, cond, detail))
    print(f"{'✅' if cond else '❌'} {name}" + (f"  ({detail})" if detail else ""))


def register(client: TestClient, username: str, invite_code: str | None = None):
    body: dict = {"username": username, "password": "pass123456"}
    if invite_code is not None:
        body["invite_code"] = invite_code
    return client.post("/api/auth/register", json=body)


def login_resp(client: TestClient, username: str):
    return client.post("/api/auth/login", json={"username": username, "password": "pass123456"})


def login_token(client: TestClient, username: str) -> str:
    r = login_resp(client, username)
    assert r.status_code == 200, f"登录失败：{r.text}"
    return r.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def role_in_db(username: str) -> str | None:
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == username).first()
        return u.role if u else None
    finally:
        db.close()


try:
    # ========== 第一轮 ==========
    # ---------- 环境①：密令未配置 ----------
    with TestClient(main.app) as c1:
        r = register(c1, "plain_user")
        check("① 无密令注册 -> user", r.status_code == 201 and r.json()["user"]["role"] == "user", f"resp={r.json()}")
        r = register(c1, "wannabe_admin", "test123")
        check("① 密令未开启时填密令 -> 400", r.status_code == 400 and "未开启" in r.json()["detail"], f"resp={r.json()}")

    # ---------- 环境③（第一轮）：create_admin.py 逻辑 = User(role="superadmin") ----------
    # 脚本交互式无法自动化，按其核心代码等价验证
    db = SessionLocal()
    try:
        db.add(User(username="root", hashed_password=hash_password("pass123456"), role="superadmin"))
        db.commit()
    finally:
        db.close()
    check("③ 超级管理员 role = superadmin", role_in_db("root") == "superadmin")

    # ---------- 环境②：.env 配 ADMIN_INVITE_CODE=test123 后重新初始化 ----------
    os.environ["ADMIN_INVITE_CODE"] = "test123"
    import config as config_mod
    importlib.reload(config_mod)  # 模拟重启进程重新读 .env
    seed_invite_code()            # 模拟重新初始化：函数内现取 config，哈希写入 site_settings

    with TestClient(main.app) as c2:
        r = register(c2, "admin_by_code", "test123")
        check("② 密令正确 -> admin", r.status_code == 201 and r.json()["user"]["role"] == "admin", f"resp={r.json()}")
        r = register(c2, "wrong_code_user", "wrong-code")
        check("② 密令错误 -> 400", r.status_code == 400 and "密令错误" in r.json()["detail"], f"resp={r.json()}")
        r = register(c2, "plain_user2")
        check("② 配好密令后不填密令 -> 仍是 user", r.status_code == 201 and r.json()["user"]["role"] == "user")

        # ----- 密令爆破限流：同 IP 连错 5 次后锁定（第 6 次 429） -----
        import auth.routes as ar
        with ar._invite_fail_lock:
            ar._invite_failures.clear()
        statuses = []
        for i in range(6):
            r = register(c2, f"brute_{i}", "bad-code")
            statuses.append(r.status_code)
        check("限流：前 5 次 400、第 6 次 429", statuses[:5] == [400] * 5 and statuses[5] == 429, f"statuses={statuses}")
        r = register(c2, "brute_final", "bad-code")
        check("限流：锁定期内继续 429", r.status_code == 429 and "10 分钟" in r.json()["detail"], f"resp={r.json()}")
        with ar._invite_fail_lock:
            ar._invite_failures.clear()  # 清掉失败记录，不影响后续用例

    # ========== 第二轮：超管专属接口 ==========
    with TestClient(main.app) as c3:
        su_token = login_token(c3, "root")
        admin_token = login_token(c3, "admin_by_code")
        user_token = login_token(c3, "plain_user")

        # ----- ① superadmin 设置密令 -> 注册填密令成为 admin -----
        r = c3.put("/api/admin/settings/invite", json={"invite_code": "newcode456"}, headers=auth(su_token))
        check("① superadmin 设置密令 -> 200", r.status_code == 200 and r.json().get("enabled") is True, f"resp={r.json()}")
        r = c3.get("/api/admin/settings/invite", headers=auth(su_token))
        check("① GET 密令状态 -> enabled=true", r.status_code == 200 and r.json()["enabled"] is True, f"resp={r.json()}")
        r = register(c3, "admin_via_settings", "newcode456")
        check("① 新密令注册 -> admin", r.status_code == 201 and r.json()["user"]["role"] == "admin", f"resp={r.json()}")

        # ----- ② admin 无权管理密令 -----
        r = c3.put("/api/admin/settings/invite", json={"invite_code": "hack"}, headers=auth(admin_token))
        check("② admin 调 PUT /settings/invite -> 403", r.status_code == 403, f"status={r.status_code}")
        r = c3.get("/api/admin/settings/invite", headers=auth(admin_token))
        check("② admin 调 GET /settings/invite -> 403", r.status_code == 403, f"status={r.status_code}")

        # ----- ③ 改角色 user→admin 生效 -----
        # 先拿 plain_user2 的 id
        r = c3.get("/api/admin/users", headers=auth(su_token))
        users_by_name = {u["username"]: u for u in r.json()["users"]}
        pid = users_by_name["plain_user2"]["id"]
        r = c3.patch(f"/api/admin/users/{pid}/role", json={"role": "admin"}, headers=auth(su_token))
        check("③ superadmin 改角色 user->admin -> 200", r.status_code == 200 and r.json()["role"] == "admin", f"resp={r.json()}")
        check("③ 改后入库角色 = admin", role_in_db("plain_user2") == "admin")
        r = c3.get("/api/admin/users", headers=auth(su_token))
        plain2 = next(u for u in r.json()["users"] if u["username"] == "plain_user2")
        check("③ 列表返回新角色", plain2["role"] == "admin")
        # 超管角色不可改（含给自己降权）
        root_id = users_by_name["root"]["id"]
        r = c3.patch(f"/api/admin/users/{root_id}/role", json={"role": "user"}, headers=auth(su_token))
        check("③ 改超管角色 -> 403", r.status_code == 403, f"status={r.status_code}")
        # admin 无权改角色
        r = c3.patch(f"/api/admin/users/{pid}/role", json={"role": "user"}, headers=auth(admin_token))
        check("③ admin 调改角色接口 -> 403", r.status_code == 403, f"status={r.status_code}")

        # ----- ④ 禁用用户后登录被拒 -----
        r = c3.patch(f"/api/admin/users/{pid}/status", json={"is_active": False}, headers=auth(admin_token))
        check("④ admin 禁用用户 -> 200", r.status_code == 200 and r.json()["is_active"] is False, f"resp={r.json()}")
        r = login_resp(c3, "plain_user2")
        check("④ 被禁用用户登录 -> 403", r.status_code == 403 and "禁用" in r.json()["detail"], f"resp={r.json()}")
        r = c3.patch(f"/api/admin/users/{pid}/status", json={"is_active": True}, headers=auth(admin_token))
        check("④ 重新启用 -> 200", r.status_code == 200 and r.json()["is_active"] is True)
        r = login_resp(c3, "plain_user2")
        check("④ 启用后登录 -> 200", r.status_code == 200, f"status={r.status_code}")
        # 不能禁用超管
        r = c3.patch(f"/api/admin/users/{root_id}/status", json={"is_active": False}, headers=auth(su_token))
        check("④ 禁用超管 -> 403", r.status_code == 403, f"status={r.status_code}")

        # ----- ⑤ 超管不能改/删其他超管 -----
        db = SessionLocal()
        try:
            db.add(User(username="root2", hashed_password=hash_password("pass123456"), role="superadmin"))
            db.commit()
        finally:
            db.close()
        r = c3.get("/api/admin/users", headers=auth(su_token))
        users_by_name = {u["username"]: u for u in r.json()["users"]}
        root2_id = users_by_name["root2"]["id"]
        r = c3.delete(f"/api/admin/users/{root2_id}", headers=auth(su_token))
        check("⑤ 超管删其他超管 -> 403", r.status_code == 403, f"status={r.status_code}")
        r = c3.patch(f"/api/admin/users/{root2_id}/role", json={"role": "admin"}, headers=auth(su_token))
        check("⑤ 超管改其他超管角色 -> 403", r.status_code == 403, f"status={r.status_code}")
        r = c3.delete(f"/api/admin/users/{root_id}", headers=auth(su_token))
        check("⑤ 超管删自己 -> 400", r.status_code == 400, f"status={r.status_code}")
        # admin 不能删 admin；superadmin 可以删 admin
        avs_id = users_by_name["admin_via_settings"]["id"]
        r = c3.delete(f"/api/admin/users/{avs_id}", headers=auth(admin_token))
        check("⑤ admin 删 admin -> 403", r.status_code == 403, f"status={r.status_code}")
        r = c3.delete(f"/api/admin/users/{avs_id}", headers=auth(su_token))
        check("⑤ 超管删 admin -> 200", r.status_code == 200, f"status={r.status_code}")

        # ----- 权限矩阵与列表统计 -----
        r = c3.get("/api/admin/users", headers=auth(su_token))
        check("④(superadmin) 列表 -> 200 且带 stats", r.status_code == 200 and "stats" in r.json(), f"resp keys={list(r.json().keys())}")
        stats = r.json()["stats"]
        users_list = r.json()["users"]
        total_ok = stats["user_count"] + stats["admin_count"] + stats["superadmin_count"] == len(users_list)
        check("stats 计数与列表一致", total_ok, f"stats={stats}")
        check("is_active 出现在列表", all("is_active" in u for u in users_list))
        r = c3.get("/api/admin/users", headers=auth(admin_token))
        check("admin 调列表 -> 200", r.status_code == 200)
        r = c3.get("/api/admin/users", headers=auth(user_token))
        check("user 调列表 -> 403", r.status_code == 403)

finally:
    engine.dispose()  # 释放 SQLite 文件句柄（Windows 下才能删库）
    TEST_DB.unlink(missing_ok=True)
    print(f"\n{'=' * 40}\n通过 {sum(1 for _, ok, _ in results if ok)}/{len(results)}")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)
