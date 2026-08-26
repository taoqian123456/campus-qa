"""
RAG 问答处理器：检索知识库 -> 组装 Prompt -> 调用大模型生成回答。

大模型不绑死单一厂商：由 config.LLM_PROVIDERS 注册表 + LLM_PROVIDER 开关决定用哪家
（见 resolve_llm）。各函数的 provider 参数可临时指定厂商，不传则用 .env 里的默认值。
"""
import os
import re
from pathlib import Path

import jieba
from openai import OpenAI

from config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    LLM_PROVIDER,
    LLM_PROVIDERS,
    MIN_SCORE,
    QUESTION_TYPES,
    THEME_KEYWORDS,
    TOP_K,
    UPLOAD_DIR,
)
from database import SessionLocal
from models import Document
from qa.vector_store import INDEX_FILE, VectorStore

SYSTEM_PROMPT = (
    "你是一个高校学生事务政策问答助手。"
    "请仅根据下面【参考资料】中的内容回答用户问题，严格遵守：\n"
    "1. 参考资料中没有与问题相关的内容时，只能回答“资料库中未找到相关信息”，严禁编造；\n"
    "2. 回答依据了某条资料时，在该句末尾标注来源编号，格式为（依据：资料N），N 是资料编号。\n"
    "回答使用中文，简洁准确，可以分点列出。\n\n"
    "【历史对话】\n"
    "{history}\n"
    "如果【历史对话】非空：当前问题里的“它/这/那/这个/那个”等指代词，请结合历史对话理解其指向"
    "（例如上一轮问过“转专业”，本轮问“那流程是什么”即指“转专业的流程”），"
    "但回答仍只依据本轮检索到的【参考资料】，历史仅用于理解指代，不引入历史里的旧结论。\n\n"
    "【参考资料】\n{context}"
)

# 历史对话里每条消息的最大长度：指代理解只需话题信息，截断防止长回答挤占上下文窗口
HISTORY_TURN_LIMIT = 100


# 指代消解式查询扩展：问题含指代词时，用历史话题词补全检索查询
_PRONOUNS = ("那", "这", "它", "其", "该", "这个", "那个", "这些", "那些", "这种", "那种", "呢")
# 提取历史话题词时过滤的常见功能词（疑问词/助词/介词等）
_QUERY_STOPWORDS = {
    "什么", "怎么", "如何", "哪些", "请问", "需要", "可以", "吗", "呢", "啊", "的", "了",
    "是", "要", "不", "会", "有", "和", "或", "在", "把", "被", "对", "从", "为", "与",
    "请", "问", "一下", "多少", "几个", "时候", "时间", "我", "你", "他", "她", "我们",
}


def _expand_query(question: str, history=None) -> str:
    """指代消解式查询扩展：问题含"那/它/这"等指代词时，把历史最后一轮问题的主题关键词
    拼进检索查询（如"那流程是什么？" -> "那流程是什么？ 转专业"），让检索能命中历史话题。

    设计要点（答辩素材）：
    - 检索只用"当前问题 + 历史话题词"，历史答案不参与检索，保持 RAG 纯净；
    - LLM 看到的用户消息仍是原始问题，配合 System Prompt 里的历史对话理解指代；
    - 无指代词时不做任何改动，避免把无关历史词灌进检索稀释相关性。
    """
    if not history or not any(p in question for p in _PRONOUNS):
        return question
    prev = (history[-1].get("question") or "").strip()
    if not prev:
        return question
    # 优先取主题关键词表里的整词（如"转专业"，BM25 命中更准，jieba 可能切成"专业"这种泛词），
    # 再用上一问的分词结果补充，最多 5 个词防止稀释
    keywords: list[str] = []
    for t in _match_themes(prev):
        for kw in THEME_KEYWORDS[t]["keywords"]:
            if kw not in keywords and len(kw) >= 2:
                keywords.append(kw)
    for w in jieba.cut(prev):
        if len(w) >= 2 and w not in _QUERY_STOPWORDS and w not in keywords:
            keywords.append(w)
    if not keywords:
        return question
    return f"{question} {' '.join(keywords[:5])}"


def _build_history_text(history=None) -> str:
    """最近 3 轮对话（用户问题 + AI 回答，各截 HISTORY_TURN_LIMIT 字）拼成文本，供 System Prompt 理解指代。"""
    if not history:
        return "（无）"
    lines = []
    for turn in history[-3:]:  # 最多带最近 3 轮，控制 token 消耗
        q = (turn.get("question") or "")[:HISTORY_TURN_LIMIT]
        a = (turn.get("answer") or "")[:HISTORY_TURN_LIMIT]
        lines.append(f"用户：{q}")
        lines.append(f"助手：{a}")
    return "\n".join(lines)


def _build_user_prompt(question: str, history=None) -> str:
    """当前问题；历史对话已拼进 System Prompt，这里保持纯净（避免双份重复）。"""
    return question


# ---------- 知识库体检：检索命中计数 ----------

# 索引 source（相对 uploads 的路径）-> documents 表原始文件名的映射缓存。
# rebuild 时 VectorStore 用 name_map 把 UUID 文件名换回原始名，这里反向建映射用于命中计数落库
_source_to_doc: dict = {}


def sync_document_registry():
    """同步索引 source 与 documents 表 / uploads 目录的对应关系。

    - 把 documents 表里没有的知识库主题文档（uploads/knowledge_base/ 下）补注册
      （status=indexed），让"知识库体检"能看到全部文档；
    - 重建 source -> document id 映射缓存，检索命中计数时按来源定位文档行。
    """
    global _source_to_doc
    db = SessionLocal()
    try:
        rows = db.query(Document).all()
        by_name = {Path(d.file_path).name: d for d in rows}
        by_filename = {d.filename: d for d in rows}
        # 已注册的纯文件名集合（去路径）：同名文件不重复补注册
        known_names = {Path(f).name for f in by_filename}

        # 补注册：磁盘上存在但 documents 表没有的知识库文档（主题文件夹里的手放文档）
        added = False
        if UPLOAD_DIR.exists():
            for p in sorted(UPLOAD_DIR.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in {".pdf", ".txt", ".docx"}:
                    continue
                if p.name in by_name:
                    continue
                rel = p.relative_to(UPLOAD_DIR).as_posix()
                if "/" not in rel:
                    continue  # 顶层 UUID 文件：documents 表该有记录；真没有说明上传流程异常，跳过
                if p.name in known_names:
                    continue  # 已有同名文档记录（如前端上传过的同名文件），不重复注册
                db.add(Document(filename=rel, file_path=str(p), status="indexed"))
                added = True
        if added:
            db.commit()
            rows = db.query(Document).all()
            by_name = {Path(d.file_path).name: d for d in rows}
            by_filename = {d.filename: d for d in rows}

        # 重建映射：索引 source（可能是原始文件名、UUID 文件名或相对路径）-> document id
        # 只存 id（int）：Document ORM 对象在 session 关闭后属性会过期
        store = get_vector_store()
        for s in dict.fromkeys(store._sources):
            key = s.rsplit("/", 1)[-1]  # 文件名部分
            doc = by_name.get(key) or by_filename.get(key) or by_filename.get(s)
            if doc is not None:
                _source_to_doc[s] = doc.id
    except Exception:
        pass  # 数据库不可用/索引为空：计数静默跳过，不影响主问答
    finally:
        db.close()


def _count_hits(results: list) -> None:
    """给本次检索命中的块所属文档累加 hit_count（每篇文档一次，不按块数重复计）。

    只写数据库、失败静默（体检是增值功能，绝不干扰主问答链路）。
    """
    if not results:
        return
    docs: set = set()
    for r in results:
        d = _source_to_doc.get(r["source"])
        if d is not None:
            docs.add(d)
    if not docs:
        return
    try:
        db = SessionLocal()
        try:
            db.query(Document).filter(Document.id.in_(docs)).update(
                {Document.hit_count: Document.hit_count + 1},
                synchronize_session=False,
            )
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


def sync_chunk_counts() -> None:
    """把索引里每篇文档的分块数同步到 documents.chunk_count（体检页展示用）。

    在 /api/admin/kb-health 被调用时刷新——检索热路径不做这件事。
    """
    try:
        store = get_vector_store()
        counts: dict[str, int] = {}
        for s in store._sources:
            counts[s] = counts.get(s, 0) + 1
        db = SessionLocal()
        try:
            for src, n in counts.items():
                doc_id = _source_to_doc.get(src)
                if doc_id is not None:
                    db.query(Document).filter(Document.id == doc_id).update(
                        {"chunk_count": n}, synchronize_session=False
                    )
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


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
        _source_to_doc.clear()  # 索引变了，旧映射作废
    return store


# ---------- 智能兜底回复（三段式：诚恳说明 + 主题引导 + 问法建议） ----------

# 问法建议模板：问题类型 -> 模板列表，按模板顺序取 1-2 条。
# {theme} 由提问命中的主题名填充（主题名自带动词，如"图书馆借阅"，模板措辞要与之自然衔接）
_SUGGEST_TEMPLATES = {
    "条件类": [
        "{theme}需要满足哪些条件？",
        "{theme}的申请资格是什么？",
        "哪些情况符合{theme}的要求？",
    ],
    "流程类": [
        "{theme}是怎么办理的？",
        "{theme}需要经过哪些步骤？",
        "{theme}在哪里申请？",
    ],
    "时间类": [
        "{theme}的办理时间是什么时候？",
        "{theme}的申请截止日期是哪天？",
        "{theme}什么时候开放申请？",
    ],
    "材料类": [
        "{theme}需要提交哪些材料？",
        "{theme}需要准备什么证明材料？",
        "{theme}的申请材料在哪里下载？",
    ],
}
_GENERIC_SUGGESTIONS = [
    "奖学金需要什么条件？",
    "转专业申请是什么流程？",
]


def _theme_doc_names(theme: str) -> list[str]:
    """从 documents 表 / 索引元数据中找主题相关文档的展示名（最多 3 个）。

    优先级：
    1. documents 表原始文件名（有"选课/奖学金"等原始命名）；
    2. 索引元数据 source（相对路径含主题文件夹名，如 "01_学籍与转专业/xxx.pdf"，覆盖知识库主题文件）；
    3. 磁盘文件名兜底（documents 记录被删但文件还在的场景）。

    每一层内再分级：文档名直接含主题关键词 > 只靠主题文件夹归属。
    仅文件夹命中会误伤同目录兄弟文档（如"02_选课与学分"下的重修细则），排后只做兜底。
    """
    cfg = THEME_KEYWORDS[theme]
    ranked: list[tuple[int, str]] = []  # (rank, 展示名)，rank 小者优先
    seen: set[str] = set()

    def add(rank: int, label: str):
        key = Path(label).stem  # 按文件名（不含后缀）去重：documents 表与索引元数据会重复命中同一文档
        if label and key not in seen:
            seen.add(key)
            ranked.append((rank, Path(label).stem))

    try:
        db = SessionLocal()
        try:
            docs = db.query(Document).all()
        finally:
            db.close()
        for d in docs:
            haystack = Path(d.filename).stem.lower()
            if any(kw in haystack for kw in cfg["keywords"]):
                add(0, d.filename)
    except Exception:
        pass  # 数据库不可用时继续走索引元数据

    # 索引元数据：source 形如 "01_学籍与转专业/xxx.pdf"，文件名命中算 1 档，仅文件夹归属算 2 档
    store = get_vector_store()
    for s in dict.fromkeys(store._sources):
        p = Path(s)
        name, dirname = p.stem.lower(), p.parent.name.lower()
        if any(kw in name for kw in cfg["keywords"]):
            add(1, p.stem)
        elif any(w in dirname for w in _theme_dirs(theme)):
            add(2, p.stem)

    # 磁盘兜底：索引元数据里没有（索引为空/未重建）时直接扫 uploads/
    if UPLOAD_DIR.exists():
        for f in sorted(UPLOAD_DIR.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in {".pdf", ".txt", ".docx"}:
                continue
            name, dirname = f.stem.lower(), f.parent.name.lower()
            if any(kw in name for kw in cfg["keywords"]):
                add(3, f.stem)
            elif any(w in dirname for w in _theme_dirs(theme)):
                add(4, f.stem)

    ranked.sort(key=lambda x: x[0])
    return [label for _, label in ranked[:3]]


# 主题文件夹名的特征词（如 "01_学籍与转专业" -> 学籍/转专业），文档名没命中关键词时用来按目录归属匹配
_THEME_DIR_WORDS = {
    "转专业": ["转专业"],
    "选课": ["选课"],
    "学分认定": ["学分认定"],  # 不含泛化词"学分"：避免"02_选课与学分"目录下所有文档（如重修细则）都归进来
    "课程重修": ["重修"],
    "奖学金": ["奖学金"],
    "毕业与学位": ["毕业", "学位"],
    "保研": ["保研", "推免"],
    "学籍管理": ["学籍"],
    "图书馆借阅": ["图书馆"],
    "英语四六级": ["四六级"],
    "上下课时间": ["上下课"],
    "在线学习": ["在线学习"],
    "学业预警": ["预警"],
}


def _theme_dirs(theme: str) -> list[str]:
    return _THEME_DIR_WORDS.get(theme, [])


def _match_themes(question: str) -> list[str]:
    """用 jieba 分词 + 主题关键词表匹配用户问题，返回命中的主题名（最多 2 个，按关键词命中数排序）。

    分词后再匹配：口语问法（"想转到计算机专业"）也能切出"转专业"关键词，
    比纯子串匹配更抗噪（词与词之间夹了语气词、标点也不会误伤）。
    """
    words = [w.lower() for w in jieba.cut(question)]
    scored = []
    for theme, cfg in THEME_KEYWORDS.items():
        count = sum(1 for w in words if w in cfg["keywords"])
        # pair_keywords：a、b 两个关键词分别「被某个分词包含」各计一次。
        # 覆盖分词把关键词切散的口语问法：jieba 把"想转到计算机专业"切成 [转到, 计算机专业]，
        # 词内包含"转"与"专业"，精确匹配会漏，包含式匹配能对上
        count += sum(
            1 for a, b in cfg.get("pair_keywords", [])
            if any(a in w for w in words) and any(b in w for w in words)
        )
        if count:
            scored.append((count, theme))
    if not scored:  # 分词未命中：整句子串兜底，保证"转专业申请时间"这类被切散的词也能归到主题
        q = question.lower()
        for theme, cfg in THEME_KEYWORDS.items():
            if any(kw in q for kw in cfg["keywords"]):
                scored.append((1, theme))
    scored.sort(key=lambda x: (-x[0], list(THEME_KEYWORDS).index(x[1])))
    return [theme for _, theme in scored[:2]]


def _question_type(question: str) -> str | None:
    """按问题类型关键词表判断问题属于条件类/流程类/时间类/材料类（第一命中的类型）。"""
    q = question.lower()
    for qtype, cfg in QUESTION_TYPES.items():
        if any(kw in q for kw in cfg["keywords"]):
            return qtype
    return None


def _build_suggestions(question: str, themes: list[str]) -> list[str]:
    """生成 1-2 个更具体的示例问法：问题类型 -> 模板，主题名填充进模板。"""
    qtype = _question_type(question)
    theme = themes[0] if themes else None
    suggests: list[str] = []
    if theme and qtype and qtype in _SUGGEST_TEMPLATES:
        for tpl in _SUGGEST_TEMPLATES[qtype][:2]:
            suggests.append(tpl.format(theme=theme))
    elif theme:
        # 类型没命中：仍给出带主题的引导问法（用第二个主题扩充到 2 条）
        if len(themes) >= 2:
            suggests = [f"{themes[0]}需要什么条件？", f"{themes[1]}怎么办理？"]
        else:
            suggests = [f"{theme}需要什么条件？", f"{theme}的办理流程是什么？"]
    else:
        suggests = _GENERIC_SUGGESTIONS[:2]
    return [s.format(theme=theme or "具体事项") for s in suggests]


def _build_graceful_reply(question: str) -> str:
    """未命中知识库时的三段式专业回复：诚恳说明 + 主题引导 + 问法建议。"""
    themes = _match_themes(question)

    lines = [
        "非常抱歉，我没有在知识库中检索到与您问题直接相关的内容，暂时无法给您准确的答复，请见谅。",
        "为尽快帮到您，我先按您的提问内容做了主题分析，您可以参考以下信息：",
    ]

    def _clean(name: str) -> str:
        # 整个列表已用《》包起来，文档名自带的书名号去掉，避免《关于印发《xx》的通知》这类嵌套
        return name.replace("《", "").replace("》", "")

    if themes:
        doc_hints: list[str] = []
        for t in themes:
            doc_hints.extend(_theme_doc_names(t))
        if doc_hints:
            names = "、".join(_clean(n) for n in dict.fromkeys(doc_hints))  # 去重保序
            lines.append(f"· 您的问题可能与「{'、'.join(themes)}」主题相关，知识库中有以下可能相关的资料：《{names}》，您可以查阅这些资料获取帮助。")
        else:
            lines.append(f"· 您的问题可能与「{'、'.join(themes)}」主题相关，建议您查阅该主题下的政策文件。")
        suggests = _build_suggestions(question, themes)
        lines.append("· 为了让我更准确地回答，建议您换一个更具体的问法，例如：" + "、".join(f"「{s}」" for s in suggests) + "。")
    else:
        suggests = _build_suggestions(question, [])
        lines.append("· 您也可以尝试换个说法提问，例如：" + "、".join(f"「{s}」" for s in suggests) + "，或先浏览知识库中的政策文件。")
    lines.append("如有其他问题，欢迎随时向我提问，我会尽力为您解答。")
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


def _retrieve_and_build(question: str, history=None, retrieval_mode: str | None = None):
    """检索 top-k 块并组装 LLM 请求 messages。

    retrieval_mode：'hybrid'/'vector'/None（None 用 config.RETRIEVAL_MODE，评测对比实验传入显式值）。

    返回 (messages, sources, empty_reply, source_map, confidence)：
    - messages：已组装好的 [system, user]；检索为空时为 None
    - sources：去重后的来源文件名列表（LLM 引用解析失败时的兜底）
    - empty_reply：检索为空时给用户的固定回复（此时 messages 为 None）
    - source_map：资料编号 -> 来源文件名（用来按 LLM 引用精确筛选来源）
    - confidence：相关度（高/中/低），基于检索原始最高相似度分数
    """
    store = get_vector_store()
    # 指代消解：问题含"那/它"等指代词时拼上历史话题词（如"那流程是什么？"->"... 转专业"）；
    # LLM 看到的仍是原始问题，历史只用于理解指代与补充检索词，不参与生成
    results = store.search(_expand_query(question, history), TOP_K, mode=retrieval_mode)

    # 相关性过滤：混合检索下，score 是向量相似度——
    # 纯 BM25 命中的块向量分记 0（未进向量 top-N），保留；向量分低于 MIN_SCORE 的弱相关块过滤
    results = [r for r in results if r["score"] == 0 or r["score"] >= MIN_SCORE]

    # 知识库体检：给本次检索命中的文档累加 hit_count（懒初始化 source->doc 映射，失败静默）
    if not _source_to_doc:
        sync_document_registry()
    _count_hits(results)

    if not results:
        raw = store.search(_expand_query(question, history), TOP_K, mode=retrieval_mode)  # 再取一次原始结果算置信度（代价小，可接受）
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
        # 历史对话拼进 System Prompt：仅用于理解"它/那"等指代；
        # 检索始终只基于当前问题（RAG 纯净），历史不参与检索、不引入旧结论
        {"role": "system", "content": SYSTEM_PROMPT.format(history=_build_history_text(history), context=context)},
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


def resolve_llm(provider: str | None = None) -> tuple[str, str, str]:
    """按厂商名从注册表解析出 (base_url, api_key, model)。

    provider 为空时用 config.LLM_PROVIDER（.env 里 LLM_PROVIDER=xxx 一行即可全局切换）。
    api_key 按注册表里的 api_key_env 现取环境变量：换厂商只需在 .env 补一行对应的 Key，
    没配 Key 的厂商也不会在启动时报错。
    """
    name = (provider or LLM_PROVIDER or "deepseek").strip()
    cfg = LLM_PROVIDERS.get(name)
    if cfg is None:
        raise ValueError(
            f"未知的模型厂商 '{name}'，可选：{'、'.join(LLM_PROVIDERS)}（在 .env 里改 LLM_PROVIDER）"
        )

    base_url, model = cfg["base_url"], cfg["model"]
    api_key = os.getenv(cfg["api_key_env"], "").strip()
    if not api_key:
        if "localhost" in base_url or "127.0.0.1" in base_url:
            api_key = "ollama"  # 本地部署不校验 Key，占位符满足 OpenAI SDK 的非空要求
        else:
            # 云端厂商缺 Key 直接报错：比等接口返回 401 更容易定位
            raise ValueError(f"{cfg['label']} 未配置 API Key，请在 backend/.env 里填 {cfg['api_key_env']}=...")
    return base_url, api_key, model


def _get_llm_client(provider: str | None = None) -> tuple[OpenAI, str]:
    """构造 OpenAI 兼容客户端，返回 (客户端, 模型名)。

    注册表里的厂商都提供 OpenAI 兼容接口，所以一个 OpenAI SDK 通吃，只换三要素。
    模型名与客户端一起返回，避免调用处再解析一次注册表。
    """
    base_url, api_key, model = resolve_llm(provider)
    return OpenAI(api_key=api_key, base_url=base_url), model


def answer_question(question: str, history=None, retrieval_mode: str | None = None, provider: str | None = None):
    """非流式 RAG 问答，返回 {"answer": str, "sources": [文件名], "confidence": 高/中/低}。

    retrieval_mode：评测对比实验用（'hybrid'/'vector'），生产走 config.RETRIEVAL_MODE。
    provider：临时指定模型厂商（'deepseek'/'zhipu'/...），不传走 config.LLM_PROVIDER。
    """
    messages, sources, empty_reply, source_map, confidence = _retrieve_and_build(
        question, history, retrieval_mode=retrieval_mode
    )

    # 检索为空时短路：省一次大模型调用，给用户三段式引导回复
    if empty_reply:
        return {"answer": empty_reply, "sources": [], "confidence": confidence, "relevant": False}

    client, model = _get_llm_client(provider)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=messages,
    )
    answer = resp.choices[0].message.content or ""
    # LLM 依据检索内容判断"没有相关内容"时，同样走三段式引导（比冷冰冰的"未找到"体验更好）
    if "未找到相关信息" in answer:
        return {"answer": _build_graceful_reply(question), "sources": [], "confidence": "低", "relevant": False}
    return {"answer": answer, "sources": _final_sources(answer, sources, source_map), "confidence": confidence, "relevant": True}


def _final_sources(answer: str, sources: list[str], source_map: dict) -> list[str]:
    """最终来源：只列回答里真正引用（依据：资料N）的文档；
    LLM 判断无相关内容时返回空；漏标引用时兜底列全部检索来源。"""
    if "未找到相关信息" in answer:
        return []
    cited = _extract_sources(answer, source_map)
    return cited or sources


def answer_question_stream(question: str, history=None, retrieval_mode: str | None = None, provider: str | None = None):
    """流式 RAG 问答（生成器），逐块 yield 事件字典：

    {"type": "token", "content": "..."}  增量文本
    {"type": "done", "answer": 完整答案, "sources": [...], "confidence": 高/中/低}  结束（含最终结果）
    {"type": "error", "message": "..."}  出错

    provider：临时指定模型厂商，不传走 config.LLM_PROVIDER。
    """
    messages, sources, empty_reply, source_map, confidence = _retrieve_and_build(
        question, history, retrieval_mode=retrieval_mode
    )

    if empty_reply:
        yield {"type": "done", "answer": empty_reply, "sources": [], "confidence": confidence, "relevant": False}
        return

    client, model = _get_llm_client(provider)
    try:
        stream = client.chat.completions.create(
            model=model,
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
        # LLM 依据检索内容判断"没有相关内容"时，同样换成三段式引导回复
        # （token 已逐块推给前端，前端 done 收尾时以这里的完整答案为准）
        if "未找到相关信息" in answer:
            answer = _build_graceful_reply(question)
            yield {"type": "done", "answer": answer, "sources": [], "confidence": "低", "relevant": False}
            return
        yield {"type": "done", "answer": answer, "sources": _final_sources(answer, sources, source_map), "confidence": confidence, "relevant": True}
    except Exception as e:  # 流已开始，只能通过事件告知前端
        yield {"type": "error", "message": f"生成回答失败：{e}"}


# ---------- 推荐问题（增值功能：失败一律静默降级，不影响主问答流程） ----------

RECOMMEND_SYSTEM_PROMPT = (
    "你是高校学生事务问答助手的追问推荐器。\n"
    "请根据【检索内容】生成 3 个用户可能接着问的问题，要求：\n"
    "1. 必须基于检索内容提问，严禁编造内容中没有的信息；\n"
    "2. 每个问题必须是短句，总长不超过 10 个字；\n"
    "3. 一行一个中文问句，不要编号、不要其他任何文字。\n\n"
    "【检索内容】\n{context}"
)


def recommend_questions(question: str, history=None, provider: str | None = None) -> list[str]:
    """基于检索到的 top-3 文档块，让大模型生成 3 个"用户可能接着问"的问题。

    provider：临时指定模型厂商，不传走 config.LLM_PROVIDER。
    任何失败（检索为空 / API 超时异常 / 返回解析不出有效问题）都返回 []，
    由调用方静默降级——推荐只是锦上添花，不影响主问答流程。
    """
    store = get_vector_store()
    results = store.search(_expand_query(question, history), TOP_K)
    results = [r for r in results if r["score"] == 0 or r["score"] >= MIN_SCORE]
    if not results:
        return []  # 检索为空（通常对应三段式兜底短路）：没有可依据的内容，不推荐

    # 主题筛选：识别提问主题，优先用主题一致的块生成推荐。
    # 防跑题：问"食堂"时检索到的奖学金/学籍块虽然分数过线，但据此生成的追问必然文不对题
    themes = _match_themes(question)
    if themes:
        keywords = set()
        for t in themes:
            keywords.update(THEME_KEYWORDS[t]["keywords"])
            keywords.update(_theme_dirs(t))
        ranked = sorted(
            results,
            key=lambda r: 0 if any(kw in r["source"].lower() for kw in keywords) else 1,
        )
        results = ranked
    # 兜底保障：过滤后 top-1 相似度仍低于低置信线，说明检索内容与问题相关性弱，
    # 基于它生成的推荐必然跑题（如问食堂却推"优秀毕业生"），直接放弃
    if results[0]["score"] != 0 and results[0]["score"] < CONFIDENCE_LOW:
        return []

    # top-3 块，每块截 200 字符控制 token（块内已含答案要点，够生成追问）
    context = "\n\n".join(f"（来源：{r['source']}）\n{r['text'][:200]}" for r in results[:3])

    try:
        client, model = _get_llm_client(provider)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.7,  # 比主问答（0.3）高：推荐需要多样性
            messages=[
                {"role": "system", "content": RECOMMEND_SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": question},
            ],
        )
    except Exception:
        return []  # API 超时/网络失败/厂商未配置 Key：静默降级

    content = resp.choices[0].message.content or ""
    # 解析：容忍编号前缀（"1. / 1、"），一行一个问句，长度 ≤ 11（10 字 + 问号），去重且不含原问题
    qs: list[str] = []
    for seg in re.split(r"[\n；;]", content):
        seg = re.sub(r"^[\d]+[\.、\)）\s\-]+", "", seg).strip().strip("「」“”\"'")
        if not (2 <= len(seg) <= 11 and seg.endswith(("？", "?"))):
            continue
        if seg in qs or seg.rstrip("？?").startswith(question.rstrip("？?")):
            continue
        qs.append(seg)
        if len(qs) >= 3:
            break
    return qs
