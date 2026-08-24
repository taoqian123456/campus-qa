"""
全局配置：数据库、上传目录、DeepSeek API 等
所有可调参数都从这里读取，改参数不用改代码。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# 加载 .env 文件（把 .env.example 复制成 .env 并填入你的配置）
load_dotenv(BASE_DIR / ".env")

APP_NAME = "高校学生事务智能问答系统"
APP_VERSION = "0.1.0"

# ---------- 数据库 ----------
# Docker 部署时用环境变量覆盖到挂载卷（如 sqlite:////data/campus_qa.db）
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'campus_qa.db').as_posix()}")

# ---------- 目录 ----------
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))        # 上传的知识库文档
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", str(BASE_DIR / "faiss_index")))  # 向量索引

UPLOAD_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# ---------- 认证（JWT 签名密钥，在 .env 里配置；正式部署务必改成随机值） ----------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ACCESS_TOKEN_EXPIRE_DAYS = 7   # JWT 有效期（天）

# ---------- DeepSeek API（在 .env 里配置） ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

# ---------- 嵌入模型（DeepSeek 无嵌入接口，默认用硅基流动 bge-m3） ----------
# 注册地址 https://siliconflow.cn ，注册后送 2000 万 token 免费额度，bge-m3 约 0.7 元/百万 token
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ---------- RAG 参数（调优实验改这里） ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))       # 分块大小（字符）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # 分块重叠
TOP_K = int(os.getenv("TOP_K", "10"))                  # 检索返回的块数
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))  # 检索过滤的低防线（bge-m3 短查询基线高，此值只挡明显垃圾；
                                                    # 真正的相关性判定靠 LLM 引用标注，见 qa_handler）

# ---------- 检索模式（论文第 6 章对比实验开关） ----------
# hybrid：BM25 关键词 + 向量语义两路召回，RRF 融合（默认，实测命中率显著高于纯向量）
# vector：纯向量检索（对比实验基线）
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

# RRF 融合权重：向量路 / BM25 路（分母 K=60 在 vector_store 里，权重为两路 score 的系数）
# 语义型问题（口语化问法）向量路权重应更高，条款型问题（关键词精确）BM25 路权重更高。
# 实测（evaluate_retrieval.py 权重扫描，67 题评测集）：本知识库以条款型政策文本为主，
# 0.4/0.6 时 Hit@5=92.5% 显著优于 0.6/0.4 的 76.1%，故默认取 BM25 路权重更高
RRF_WEIGHTS = {
    "vector": float(os.getenv("RRF_W_VECTOR", "0.4")),
    "bm25": float(os.getenv("RRF_W_BM25", "0.6")),
}

# ---------- 置信度阈值（回答的"相关度：高/中/低"判定，实验后可调） ----------
CONFIDENCE_HIGH = float(os.getenv("CONFIDENCE_HIGH", "0.6"))
CONFIDENCE_LOW = float(os.getenv("CONFIDENCE_LOW", "0.4"))

# ---------- 主题关键词表（智能兜底：未命中时引导用户到相关主题） ----------
# 每个主题：keywords 用于匹配用户问题与文档名（jieba 分词后按词匹配），
# suggestions 是给用户的示例问法（按问题类型生成问法建议失败时的兜底）
# 主题划分与 uploads/knowledge_base/ 下的主题文件夹对应，答辩时可讲"主题引导机制"
THEME_KEYWORDS = {
    "转专业": {
        # "转"/"专业"成对出现算一次命中（口语问法"想转到计算机专业"分词后是 [转, 专业]），
        # 单独出现不计入，避免"学分转换"之类误归到转专业
        "keywords": ["转专业", "专业调整", "转专业申请", "转出", "转入"],
        "pair_keywords": [("转", "专业")],
        "suggestions": ["转专业需要什么条件？", "转专业申请在什么时间？"],
    },
    "选课": {
        "keywords": ["选课", "公选课", "选修课", "公共选修课", "课表", "通识教育选修"],
        "suggestions": ["公共选修课怎么选课？", "选课时间安排是什么？"],
    },
    "学分认定": {
        "keywords": ["学分", "认定", "转换", "互认", "绩点"],
        "suggestions": ["学分认定和转换什么时候办理？", "什么课程可以互认学分？"],
    },
    "课程重修": {
        "keywords": ["重修", "补考", "挂科", "不及格"],
        "suggestions": ["哪些课程必须重修？", "重修申请什么时候办理？"],
    },
    "奖学金": {
        "keywords": ["奖学金", "评优", "三好学生", "优秀毕业生", "助学金", "励志"],
        "suggestions": ["一等奖学金需要什么条件？", "优秀毕业生怎么评选？"],
    },
    "毕业与学位": {
        "keywords": ["毕业", "学位", "学士", "证书", "毕业证", "结业", "肄业", "学位证", "gpa"],
        "suggestions": ["授予学士学位有什么要求？", "毕业证丢了怎么办？"],
    },
    "保研": {
        "keywords": ["保研", "推免", "推免生", "研究生推荐", "免试研究生"],
        "suggestions": ["保研需要什么条件？", "推免资格怎么申请？"],
    },
    "学籍管理": {
        "keywords": ["学籍", "注册", "休学", "复学", "退学", "入学", "报到", "保留入学资格", "请假"],
        "suggestions": ["新生报到有什么要求？", "保留入学资格的条件是什么？"],
    },
    "图书馆借阅": {
        "keywords": ["图书馆", "借书", "借阅", "还书", "委托借阅", "读者证"],
        "suggestions": ["惠州校区怎么借书？", "可以委托借其他校区的书吗？"],
    },
    "英语四六级": {
        "keywords": ["四六级", "cet", "英语四级", "英语六级", "六级"],
        "suggestions": ["CET 报名流程是怎样的？", "CET6 资格怎么复核？"],
    },
    "上下课时间": {
        "keywords": ["上下课", "上课时间", "下课时间", "作息时间", "教学时间"],
        "suggestions": ["本学期上下课时间是怎么安排的？", "各教学楼的作息时间一样吗？"],
    },
    "在线学习": {
        "keywords": ["在线学习", "网络教学", "网课", "线上学习", "学习平台"],
        "suggestions": ["在线学习平台怎么使用？", "网络课程如何登录学习？"],
    },
    "学业预警": {
        "keywords": ["预警", "旷课", "考勤"],
        "suggestions": ["什么情况会触发学业预警？", "考勤预警的标准是什么？"],
    },
}

# ---------- 问题类型关键词表（智能兜底：按问题类型生成更具体的问法建议） ----------
# 检索未命中时，先按关键词判断问题属于哪类，再套用对应问法模板，
# 让"问法建议"从写死文案升级为"问题类型 -> 模板"的生成式机制（答辩素材）
QUESTION_TYPES = {
    # 顺序即优先级（"转专业申请时间是什么时候"里"申请"和"时间"并存，应判为时间类）
    "时间类": {"keywords": ["时间", "什么时候", "何时", "截止", "期限", "哪天", "几点"]},
    "条件类": {"keywords": ["条件", "要求", "资格", "标准", "门槛", "可以吗", "能不能", "能吗", "吗"]},
    "材料类": {"keywords": ["材料", "证明", "文件", "表格", "证件", "需要带", "提交什么", "交什么"]},
    "流程类": {"keywords": ["流程", "步骤", "怎么", "如何", "申请", "办理", "操作", "报名"]},
}

# ---------- OCR（扫描版 PDF 兜底，可选） ----------
# 开启后，pypdf 提取为空的 PDF 会渲染成图片走 RapidOCR 识别（首次初始化需联网下载模型，约 15MB）
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") == "1"
