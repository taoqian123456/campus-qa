"""
嵌入模型工厂：langchain-openai 的 OpenAIEmbeddings 对接硅基流动等 OpenAI 兼容接口。
"""
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL


def get_embedder() -> OpenAIEmbeddings:
    """按 config 配置构造嵌入器（每次新建，OpenAIEmbeddings 内部有 HTTP 连接复用）。

    当前默认走硅基流动 BAAI/bge-m3（DeepSeek 没有嵌入接口）。
    """
    if not EMBEDDING_API_KEY:
        raise RuntimeError(
            "EMBEDDING_API_KEY 未配置：请在 .env 里填入硅基流动的 API Key"
            "（注册 https://siliconflow.cn -> 账户管理 -> API 密钥）"
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
    )
