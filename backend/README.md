# 高校学生事务智能问答系统（backend）

毕业设计《基于大语言模型与向量检索的高校学生事务智能问答系统设计与实现》的后端项目。
FastAPI 服务，包含：注册登录、RAG 问答（流式 + 来源标注）、会话管理、管理后台、多模型可插拔。

## 目录结构

```
backend/
├── main.py            # FastAPI 入口（/health、/docs、前端页面）
├── config.py          # 全局配置（数据库、多模型注册表、RAG 参数）
├── database.py        # 数据库引擎与会话
├── models.py          # 4 张表：users / sessions / messages / documents
├── init_db.py         # 运行它来建表/迁移
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
   `.env` 里还有 `LLM_PROVIDER`（默认对话模型开关）和其他平台 key 的占位符，
   只用 DeepSeek 的话保持 `LLM_PROVIDER=deepseek` 不动即可，其他占位行不用管
   （详见下文"多模型支持"）。
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

## 多模型支持

系统不绑死单一模型厂商：RAG 检索链路与"用哪家模型生成回答"解耦，
在 `.env` 里改一行即可全局切换，前端顶栏下拉框还能按需临时切换（每条回答会标注所用模型）。

### 支持列表

| 模型 | 平台 | 注册地址 | 免费情况 |
|---|---|---|---|
| DeepSeek | 深度求索 | https://platform.deepseek.com | 充值后按量计费（毕设 10-20 元足够） |
| 智谱 GLM | 智谱开放平台 | https://open.bigmodel.cn | glm-4-flash 免费 |
| 阿里通义 | 阿里百炼 | https://bailian.console.aliyun.com | 新用户有免费额度，之后按量计费 |
| Kimi | 月之暗面 | https://platform.moonshot.cn | 按量计费 |
| 硅基流动 | SiliconFlow | https://siliconflow.cn | 注册送额度（嵌入模型同平台） |
| 本地 Ollama | 本地部署 | https://ollama.com | 完全免费（用你自己的显卡/CPU） |

### 使用方法

**方式一：默认模型全局切换（推荐）**

1. 复制 `.env.example` 为 `.env`（若还没有）；
2. 在 `.env` 里填入对应平台的 API Key（如 `ZHIPU_API_KEY=sk-你的key`）；
3. 设置 `LLM_PROVIDER=zhipu`（可选值见 .env.example 注释）；
4. 重启服务，之后所有问答默认走该模型。

**方式二：前端下拉框切换**

登录后在顶栏选择模型，切换立即生效（仅影响之后的提问，历史回答标注各自所用模型），
选择会保存在浏览器本地，刷新页面不丢失。

> **密钥安全声明**：本项目只存储和使用您自己的 API Key。密钥只存在于您本机的 `.env`
> 文件中，不会上传到任何服务器、不会写入数据库、不会进入 Docker 镜像（.env 已加入
> .gitignore 与 .dockerignore）。


## 常见问题

| 问题 | 解决 |
|---|---|
| `uvicorn` 不是内部或外部命令 | 没激活虚拟环境，重新执行 `venv\Scripts\activate` |
| 端口被占用（8000 已被用） | 改用 `uvicorn main:app --reload --port 8001` |
| 报错 `bcrypt __about__` | 执行 `pip install bcrypt==4.0.1` |
| 改了代码没生效 | 确认启动命令带 `--reload`，或手动重启服务 |
| 忘记 .env 配置 | 后端会报 API Key 相关错误，检查 backend/.env 是否存在且填了 Key |
| 切换模型后报错 | 检查三点：① 对应平台的 Key 是否填对（`.env` 里 `XXX_API_KEY`）；② 该平台是否已开通/有额度；③ `LLM_PROVIDER` 拼写是否与 .env.example 注释里的可选值一致（错了后端会提示可选列表） |
| Ollama 连不上 | 先确认本机已启动 `ollama serve` 且执行过 `ollama pull qwen2.5:7b`；再检查 `.env` 里 `OLLAMA_BASE_URL` 端口是否正确（默认 `http://localhost:11434/v1`） |
