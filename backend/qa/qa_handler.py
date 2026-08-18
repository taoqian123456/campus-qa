"""
RAG 问答处理器：检索知识库 -> 组装 Prompt -> 调用 DeepSeek 生成回答。
"""
import re

from openai import OpenAI

from config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    MIN_SCORE,
    THEME_KEYWORDS,
    TOP_K,
    UPLOAD_DIR,
)
from qa.vector_store import INDEX_FILE, VectorStore

SYSTEM_PROMPT = (
    "你是一个高校学生事务政策问答助手。"
    "请仅根据下面【参考资料】中的内容回答用户问题，严格遵守：\n"
    "1. 参考资料中没有与问题相关的内容时，只能回答“资料库中未找到相关信息”，严禁编造；\n"
    "2. 回答依据了某条资料时，在该句末尾标注来源编号，格式为（依据：资料N），N 是资料编号。\n"
    "回答使用中文，简洁准确，可以分点列出。\n\n"
    "【参考资料】\n{context}"
)

# 惰性单例：每次调用检查 index.faiss 的修改时间，rebuild 之后无需重启服务
_store_cache: dict = {}


def get_vector_store() -> VectorStore:
    """获取共享的 VectorStore；索引文件被 rebuild 后自动重新加载。"""
    mtime = INDEX_FILE.stat().st_mtime if INDEX_FILE.exists() else None
    store, cached_mtime = _store_cache.get("store"), _store_cache.get("mtime")
    if store is None or mtime != cached_mtime:
        store = VectorStore()
        _store_cache["store"] = store
        _store_cache["mtime"] = mtime
    return store


def _build_user_prompt(question: str, history=None) -> str:
    """当前问题 + 最近几轮对话（history: [{"question": ..., "answer": ...}, ...]）。"""
    if not history:
        return question
    lines = ["【历史对话】"]
    for turn in history[-3:]:  # 最多带最近 3 轮，控制 token 消耗
        lines.append(f"用户：{turn.get('question', '')}")
        lines.append(f"助手：{turn.get('answer', '')}")
    lines.append(f"【当前问题】\n{question}")
    return "\n".join(lines)


def _match_themes(question: str) -> list[str]:
    """用主题关键词表匹配用户问题，返回命中的主题名（最多 2 个，保持表顺序）。"""
    q = question.lower()
    hits = []
    for theme, cfg in THEME_KEYWORDS.items():
        if any(kw in q for kw in cfg["keywords"]):
            hits.append(theme)
        if len(hits) >= 2:
            break
    return hits


def _theme_docs(theme: str) -> list[str]:
    """从知识库目录里找与主题相关的文档名（父文件夹名 + 文件名都参与匹配，最多 3 个）。"""
    cfg = THEME_KEYWORDS[theme]
    docs: list[str] = []
    if UPLOAD_DIR.exists():
        for p in sorted(UPLOAD_DIR.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in {".pdf", ".txt", ".docx"}:
                continue
            haystack = (p.parent.name + p.stem).lower()
            if any(kw in haystack for kw in cfg["keywords"]):
                docs.append(p.stem)
            if len(docs) >= 3:
                break
    return docs


def _build_graceful_reply(question: str) -> str:
    """未命中知识库时的三段式专业回复：诚恳说明 + 主题引导 + 问法建议。"""
    themes = _match_themes(question)

    lines = ["抱歉，在现有知识库中没有检索到与您问题直接相关的内容。"]
    if themes:
        doc_hints: list[str] = []
        for t in themes:
            doc_hints.extend(_theme_docs(t))
        if doc_hints:
            names = "、".join(dict.fromkeys(doc_hints))  # 去重保序
            lines.append(f"不过，根据知识库内容，与您的问题最相关的主题有：{'、'.join(themes)}，"
                         f"建议您从《{names}》等资料中查询。")
        else:
            lines.append(f"不过，根据知识库内容，与您的问题最相关的主题有：{'、'.join(themes)}。")
        suggests: list[str] = []
        for t in themes:
            suggests.extend(THEME_KEYWORDS[t]["suggestions"])
        lines.append("您也可以尝试更具体的问法，例如：" + "、".join(f"「{s}」" for s in suggests[:2]) + "。")
    else:
        lines.append("您可以尝试换个说法提问，或先浏览知识库中的政策文件（如《学生手册》《学籍管理办法》等）。")
    return "\n".join(lines)


def _confidence_of(raw_results: list) -> str:
    """根据检索原始最高分给出相关度：高 / 中 / 低。"""
    if not raw_results:
        return "低"
    top = max(r["score"] for r in raw_results)
    if top >= CONFIDENCE_HIGH:
        return "高"
    if top >= CONFIDENCE_LOW:
        return "中"
    return "低"


def _retrieve_and_build(question: str, history=None):
    """检索 top-k 块并组装 LLM 请求 messages。

    返回 (messages, sources, empty_reply, source_map, confidence)：
    - messages：已组装好的 [system, user]；检索为空时为 None
    - sources：去重后的来源文件名列表（LLM 引用解析失败时的兜底）
    - empty_reply：检索为空时给用户的固定回复（此时 messages 为 None）
    - source_map：资料编号 -> 来源文件名（用来按 LLM 引用精确筛选来源）
    - confidence：相关度（高/中/低），基于检索原始最高相似度分数
    """
    store = get_vector_store()
    results = store.search(question, TOP_K)

    # 相关性过滤：分数低于 MIN_SCORE 的块不进上下文、不列来源
    results = [r for r in results if r["score"] >= MIN_SCORE]

    if not results:
        raw = store.search(question, TOP_K)  # 再取一次原始结果算置信度（代价小，可接受）
        empty_reply = _build_graceful_reply(question)
        return None, [], empty_reply, {}, _confidence_of(raw)

    # 参考资料按检索顺序编号，并标注来源文件
    blocks = []
    source_map = {}
    for i, r in enumerate(results, 1):
        blocks.append(f"【资料{i}】（来源：{r['source']}）\n{r['text']}")
        source_map[str(i)] = r["source"]
    context = "\n\n".join(blocks)

    # sources：来自检索结果，去重并保持顺序（LLM 引用解析失败时兜底）
    sources = list(dict.fromkeys(r["source"] for r in results))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": _build_user_prompt(question, history)},
    ]
    return messages, sources, None, source_map, _confidence_of(results)


_CITE_RE = re.compile(r"（依据：资料(\d+)）|\[(\d+)\]")


def _extract_sources(answer: str, source_map: dict) -> list[str]:
    """从 LLM 回答里解析引用标注（依据：资料N），只保留真正被引用的来源文件名。"""
    cited: list[str] = []
    for m in _CITE_RE.finditer(answer):
        n = m.group(1) or m.group(2)
        src = source_map.get(n)
        if src and src not in cited:
            cited.append(src)
    return cited


def _get_llm_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def answer_question(question: str, history=None):
    """非流式 RAG 问答，返回 {"answer": str, "sources": [文件名], "confidence": 高/中/低}。"""
    messages, sources, empty_reply, source_map, confidence = _retrieve_and_build(question, history)

    # 检索为空时短路：省一次大模型调用，给用户三段式引导回复
    if empty_reply:
        return {"answer": empty_reply, "sources": [], "confidence": confidence}

    client = _get_llm_client()
    resp = client.chat.completions.create(
        model=DEEPSEEK_CHAT_MODEL,
        temperature=0.3,
        messages=messages,
    )
    answer = resp.choices[0].message.content or ""
    # LLM 依据检索内容判断"没有相关内容"时，同样走三段式引导（比冷冰冰的"未找到"体验更好）
    if "未找到相关信息" in answer:
        return {"answer": _build_graceful_reply(question), "sources": [], "confidence": "低"}
    return {"answer": answer, "sources": _final_sources(answer, sources, source_map), "confidence": confidence}


def _final_sources(answer: str, sources: list[str], source_map: dict) -> list[str]:
    """最终来源：只列回答里真正引用（依据：资料N）的文档；
    LLM 判断无相关内容时返回空；漏标引用时兜底列全部检索来源。"""
    if "未找到相关信息" in answer:
        return []
    cited = _extract_sources(answer, source_map)
    return cited or sources


def answer_question_stream(question: str, history=None):
    """流式 RAG 问答（生成器），逐块 yield 事件字典：

    {"type": "token", "content": "..."}  增量文本
    {"type": "done", "answer": 完整答案, "sources": [...], "confidence": 高/中/低}  结束（含最终结果）
    {"type": "error", "message": "..."}  出错
    """
    messages, sources, empty_reply, source_map, confidence = _retrieve_and_build(question, history)

    if empty_reply:
        yield {"type": "done", "answer": empty_reply, "sources": [], "confidence": confidence}
        return

    client = _get_llm_client()
    try:
        stream = client.chat.completions.create(
            model=DEEPSEEK_CHAT_MODEL,
            temperature=0.3,
            messages=messages,
            stream=True,
        )
        parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
                yield {"type": "token", "content": delta}
        answer = "".join(parts)
        yield {"type": "done", "answer": answer, "sources": _final_sources(answer, sources, source_map), "confidence": confidence}
    except Exception as e:  # 流已开始，只能通过事件告知前端
        yield {"type": "error", "message": f"生成回答失败：{e}"}
