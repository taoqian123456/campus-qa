# 高校学生事务智能问答系统（backend 骨架）

毕业设计《基于大语言模型与向量检索的高校学生事务智能问答系统设计与实现》的后端项目。
当前是**可运行的骨架**：FastAPI 服务 + 数据库 4 张表。后续功能（注册登录、RAG 问答、会话、管理后台）按 `prompts/提示词清单.md` 用 AI 逐步生成。

## 目录结构

```
backend/
├── main.py            # FastAPI 入口（现在有 /health 和 /docs）
├── config.py          # 全局配置（数据库、DeepSeek API、RAG 参数）
├── database.py        # 数据库引擎与会话
├── models.py          # 4 张表：users / sessions / messages / documents
├── init_db.py         # 运行它来建表
├── requirements.txt   # 依赖清单
├── .env.example       # 配置模板（复制为 .env 后填入 Key）
├── uploads/           # 知识库文档目录（自动创建）
└── faiss_index/       # 向量索引目录（自动创建）
```

## 第一次启动（约 20 分钟，照做即可）

1. **装 Python**（若已装跳过）：python.org 下载 Python 3.10+，安装时**勾选 Add Python to PATH**。
2. **打开项目**：VSCode → 文件 → 打开文件夹 → 选择 `backend` 文件夹。
3. **建虚拟环境 + 装依赖**：VSCode 里按 `` Ctrl+` `` 打开终端，依次执行：
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   （pip 太慢可先执行：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`）
4. **配置 API Key**：把 `.env.example` 复制一份改名为 `.env`，打开填入你的 DeepSeek API Key。
5. **初始化数据库**：
   ```bash
   python init_db.py
   ```
   看到 `✅ 数据库初始化完成` 即成功。
6. **启动服务**：
   ```bash
   uvicorn main:app --reload
   ```
7. **验证**（两个都要看到）：
   - 浏览器打开 http://127.0.0.1:8000/health → 显示 `{"status":"ok",...}`
   - 浏览器打开 http://127.0.0.1:8000/docs → 出现 Swagger API 文档页面（以后每个接口都能在这里测试）

## 下一步

打开 `prompts/提示词清单.md`，从 **P1 用户认证** 开始，按天推进。
每次让 AI 写完代码后，到这里跑 `uvicorn main:app --reload` 并到 /docs 测试验收。

## 常见问题

| 问题 | 解决 |
|---|---|
| `uvicorn` 不是内部或外部命令 | 没激活虚拟环境，重新执行 `venv\Scripts\activate` |
| 端口被占用（8000 已被用） | 改用 `uvicorn main:app --reload --port 8001` |
| 报错 `bcrypt __about__` | 执行 `pip install bcrypt==4.0.1` |
| 改了代码没生效 | 确认启动命令带 `--reload`，或手动重启服务 |
| 忘记 .env 配置 | 后端会报 API Key 相关错误，检查 backend/.env 是否存在且填了 Key |
