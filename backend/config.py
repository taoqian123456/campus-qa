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
DATABASE_URL = f"sqlite:///{(BASE_DIR / 'campus_qa.db').as_posix()}"

# ---------- 目录 ----------
UPLOAD_DIR = BASE_DIR / "uploads"        # 上传的知识库文档
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"  # 向量索引

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

# ---------- 置信度阈值（回答的"相关度：高/中/低"判定，实验后可调） ----------
CONFIDENCE_HIGH = float(os.getenv("CONFIDENCE_HIGH", "0.6"))
CONFIDENCE_LOW = float(os.getenv("CONFIDENCE_LOW", "0.4"))

# ---------- 主题关键词表（智能兜底：未命中时引导用户到相关主题） ----------
# 每个主题：keywords 用于匹配用户问题与文档名，suggestions 是给用户的示例问法
THEME_KEYWORDS = {
    "转专业": {
        "keywords": ["转专业", "专业调整", "转专业申请"],
        "suggestions": ["转专业需要什么条件？", "转专业申请在什么时间？"],
    },
    "学籍管理": {
        "keywords": ["学籍", "注册", "休学", "复学", "退学", "入学", "报到", "保留入学资格", "请假"],
        "suggestions": ["新生报到有什么要求？", "保留入学资格的条件是什么？"],
    },
    "奖学金": {
        "keywords": ["奖学金", "评优", "三好学生", "优秀毕业生"],
        "suggestions": ["一等奖学金需要什么条件？", "优秀毕业生怎么评选？"],
    },
    "学分认定": {
        "keywords": ["学分", "认定", "转换", "互认", "绩点"],
        "suggestions": ["学分认定和转换什么时候办理？", "什么课程可以互认学分？"],
    },
    "课程重修": {
        "keywords": ["重修", "补考", "挂科", "不及格"],
        "suggestions": ["哪些课程必须重修？", "重修申请什么时候办理？"],
    },
    "学业预警": {
        "keywords": ["预警", "旷课", "考勤"],
        "suggestions": ["什么情况会触发学业预警？", "考勤预警的标准是什么？"],
    },
    "学士学位": {
        "keywords": ["学位", "学士", "gpa"],
        "suggestions": ["授予学士学位有什么要求？", "GPA 低于 2.0 还能拿学位吗？"],
    },
    "学业证书": {
        "keywords": ["证书", "毕业证", "结业", "肄业", "学位证"],
        "suggestions": ["学业证书包括哪些？", "毕业证丢了怎么办？"],
    },
    "英语四六级": {
        "keywords": ["四六级", "cet", "英语四级", "英语六级"],
        "suggestions": ["CET 报名流程是怎样的？", "CET6 资格怎么复核？"],
    },
    "图书馆借阅": {
        "keywords": ["图书馆", "借书", "借阅", "还书", "委托借阅"],
        "suggestions": ["惠州校区怎么借书？", "可以委托借其他校区的书吗？"],
    },
}

# ---------- OCR（扫描版 PDF 兜底，可选） ----------
# 开启后，pypdf 提取为空的 PDF 会渲染成图片走 RapidOCR 识别（首次初始化需联网下载模型，约 15MB）
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") == "1"
