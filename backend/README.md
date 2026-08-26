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

## 下一步：评测与论文实验

系统功能已开发完成（认证 / RAG 问答 / 会话 / 管理后台 / 多模型），实验数据用以下脚本产出：

- **论文第 6 章对比实验**：67 题评测集（`backend/eval_set.json`）跑不同参数/模型后人工打分。
  常用命令（在 backend/ 目录、虚拟环境激活状态下）：
  ```bash
  python evaluate.py --chunk_size 400 --top_k 5 --mode hybrid              # 参数对比（会重建索引）
  python evaluate.py --top_k 8 --mode hybrid --provider zhipu              # 多模型对比（不重建索引）
  python evaluate_retrieval.py --top_k 5                                   # 检索命中率 Hit@k（不调 LLM）
  python perf_test.py --provider deepseek                                  # 问答耗时
  ```
  生成的 CSV 可用 `python csv_to_xlsx.py` 转成 xlsx（宽列 + 表头样式），在 Excel/WPS 里
  给"相关性/忠实度"打分。跑完记得用管理后台"重建索引"恢复正式参数（.env：c400/k8/混合）。
- **答辩演示**：`uvicorn main:app --reload` 启动后访问 http://127.0.0.1:8000/ ，
  注册一个账号即可演示完整流程（提问 → 来源标注 → 追问推荐 → 模型切换 → 管理后台）。

## Docker 部署

> 目标：**任何人有 Docker 就能一键部署运行**——不需要装 Python、不需要装依赖、不挑操作系统。
> 文件清单（项目根目录）：`Dockerfile`、`docker-compose.yml`、`start-docker.bat`（Windows）、`start-docker.sh`（Linux/Mac）、`.dockerignore`

### 对使用者的要求

- 安装 **Docker Desktop**（官网 https://www.docker.com/products/docker-desktop/ ；Linux 装 docker + docker compose 插件）
- 至少配置**一个对话模型 Key**（推荐 DeepSeek）+ **硅基流动 Key**（嵌入模型，必须）

### 部署步骤（3 步）

**第 1 步：拿到项目文件夹** —— 把整个 `campus-qa` 文件夹拷给对方（或用 git 仓库）。

**第 2 步：填 API Key** —— 打开 `backend\.env`（没有就把 `backend\.env.example` 复制成 `.env`）：

必填（系统运行必须）：
```ini
DEEPSEEK_API_KEY=sk-你的DeepSeekKey
EMBEDDING_API_KEY=sk-你的硅基流动Key   # 嵌入模型，向量检索用，必须
SECRET_KEY=随便写一串随机字符
```

选填（多模型，想用哪个填哪个，**不填就不显示在前端下拉框**）：
```ini
LLM_PROVIDER=deepseek                # 默认对话模型：deepseek / zhipu / dashscope / moonshot / siliconflow / ollama
ZHIPU_API_KEY=sk-你的智谱key         # 智谱 GLM（glm-4-flash 免费）
DASHSCOPE_API_KEY=sk-你的通义key     # 阿里通义 qwen-plus
MOONSHOT_API_KEY=sk-你的kimi key     # Kimi
SILICONFLOW_API_KEY=sk-你的硅基流动key  # 硅基流动上的其他模型
```

> 🔒 **隐私安全**：Key 只写在本机 `.env` 里，通过 docker-compose 的 env_file 注入容器——
> Key 不会进入镜像、不会入库、不会出现在任何代码和文档中。本项目只使用你自己的 Key，不收集不上传。

**第 3 步：启动** —— Windows 双击 `start-docker.bat`；Linux/Mac 执行 `./start-docker.sh`。
首次构建约 5-10 分钟（下载镜像+依赖），之后启动秒开。完成后浏览器打开 **http://localhost:8000** 即可使用。

### 多模型切换怎么用（部署后）

1. 登录后**顶栏有模型下拉框**，直接切换 DeepSeek / 智谱 / 通义等（前提是 .env 里填了对应 Key）
2. 每条 AI 回答下方显示"由 xx 生成"，方便对比不同模型效果
3. 想固定默认模型：改 `.env` 里 `LLM_PROVIDER=zhipu` 一行 → 重启容器

> ⚠️ **Ollama 特殊说明**：Ollama 是跑在宿主机上的本地模型，容器内无法用 `localhost` 访问宿主机。
> 要在 Docker 部署下用 Ollama，`.env` 里写 `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`
> （Linux 还需在 compose 加 `extra_hosts: host.docker.internal:host-gateway`）。普通用户直接用云端模型即可。

### 知识库文档怎么进系统

1. **网页上传（推荐）**：注册管理员 → 管理后台 → 知识库管理 → 上传 PDF/TXT/DOCX → 点"重建索引"
2. **放进数据卷**：`docker cp 你的文档.pdf campus-qa:/data/uploads/` 后在管理后台点"重建索引"
3. **项目自带知识库**：把文档放在 `backend\uploads\knowledge_base\` 再构建镜像（不建议——更新文档要重建镜像）

> ⚠️ 上传扫描版 PDF 时，容器首次 OCR 需要联网下载识别模型（约 15MB），之后有缓存。

### 数据与升级

| 事项 | 说明 |
|---|---|
| 数据存在哪 | Docker 卷 `campus_data`（SQLite + uploads + 索引），容器删了数据还在 |
| 备份数据 | `docker run --rm -v campus_data:/data -v %cd%:/backup alpine tar czf /backup/data-backup.tar.gz /data`（Windows 用 %cd%） |
| 升级代码 | 重新 `docker compose up -d --build`，数据不丢 |
| 停用/启用 | `docker compose down` / `docker compose up -d` |
| 看日志 | `docker compose logs -f` |

### 部署常见问题

| 问题 | 解决 |
|---|---|
| 启动后页面打不开 | `docker compose logs` 看日志；确认 8000 端口没被占用（占用就改 docker-compose.yml 的 `"8000:8000"` 为 `"8001:8000"`） |
| 报 API Key 错误 | 检查 backend/.env 的 key 是否正确、是否在 compose 构建**之后**才改的（改了要 `docker compose up -d` 重启容器） |
| 问答报嵌入错误 | 硅基流动 key 余额/额度问题，去 https://siliconflow.cn 检查 |
| 重建索引很慢 | 正常：扫描版 PDF 要逐页 OCR，耐心等 |
| 想换端口 | 改 docker-compose.yml 的 ports 映射 |

### 部署的用处（论文/答辩素材）

1. **论文第 5 章**：系统部署一节可写"基于 Docker 容器化部署，环境无关、一键启动"，附 docker-compose.yml 核心配置
2. **答辩演示**：换一台没装过 Python 的电脑，装 Docker 就能现场跑起来
3. **简历**："通过 Docker 容器化封装部署，实现环境无关的一键部署"是工程能力加分项
4. **给导师/同学试用**：把 campus-qa 文件夹发过去，3 步跑起来

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
