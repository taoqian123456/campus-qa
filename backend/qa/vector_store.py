"""
FAISS 向量索引 + BM25 关键词索引（混合检索）：把知识库分块向量化、持久化到 faiss_index/ 目录。

混合检索动机（答辩素材）：
- 实测 bge-m3 对"口语化问题 vs 书面政策条款"的向量区分度不足（语义无关块与相关块分数接近）；
- BM25 关键词检索（rank_bm25 + jieba 分词）对条款型文本（报到/请假/奖学金等关键词）命中精准；
- 两者通过加权 RRF（Reciprocal Rank Fusion）融合：score = w_vec/(K+rank_vec) + w_bm25/(K+rank_bm25)，
  权重在 config.RRF_WEIGHTS 可调，检索模式在 config.RETRIEVAL_MODE（hybrid/vector）可切换，
  用于论文第 6 章对比实验。

存储结构：
  faiss_index/index.faiss   向量（IndexFlatL2，向量已做 L2 归一化，平方距离/2 = 1 - 余弦相似度）
  faiss_index/index.pkl     元数据：块文本、来源文件名、块顺序、分词后的块（BM25Okapi 用）
"""
import pickle
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

import config  # 动态读取 config.RETRIEVAL_MODE / RRF_WEIGHTS（评测脚本覆盖后即时生效）
from qa.embeddings import get_embedder
from qa.knowledge_base import chunk_text, extract_text

INDEX_FILE = config.FAISS_INDEX_DIR / "index.faiss"
META_FILE = config.FAISS_INDEX_DIR / "index.pkl"

# 扫描时跳过的临时/系统文件；~$ 开头是 Office 锁文件，._ 开头是 macOS AppleDouble
TEMP_FILENAMES = {".ds_store", "thumbs.db", "desktop.ini"}
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx"}  # 与 admin.documents.ALLOWED_SUFFIXES 保持一致

# 混合检索参数
RRF_K = 60      # RRF 分母常数（越大融合对排名越不敏感）
VEC_TOP_N = 50  # 向量路召回数（融合用）
BM25_TOP_N = 50  # BM25 路召回数（融合用）


def _tokenize(text: str) -> list[str]:
    """jieba 分词 + 轻清洗：去单字、纯数字、空白（BM25 与查询共用）。"""
    return [w for w in jieba.cut(text) if len(w.strip()) > 1 and not w.strip().isdigit()]


def _iter_upload_documents(upload_dir: Path):
    """递归列出 upload_dir 下所有可索引文档（*.pdf/*.txt/*.docx），按路径排序。

    跳过临时文件（.DS_Store、Thumbs.db、~$ 锁文件等）；后缀不支持的文件静默跳过。
    """
    for path in sorted(upload_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in TEMP_FILENAMES or name.startswith("~$") or name.startswith("._"):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        yield path


class VectorStore:
    """FAISS 索引的封装：加载/添加/搜索/保存/重建。

    元数据列表与 faiss 索引行一一对应：
      faiss 第 i 行  <->  self._texts[i] / self._sources[i] / self._tokenized[i]
    顺序（order）留给未来「按文档内顺序拼接上下文」用，暂未使用。
    """

    def __init__(self, embedder=None):
        """加载已有索引；不存在则建空索引。"""
        self.embedder = embedder or get_embedder()
        self.index: faiss.Index | None = None
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._orders: list[int] = []
        self._tokenized: list[list[str]] = []   # 每块的分词结果（BM25Okapi 语料）
        self._bm25: BM25Okapi | None = None
        self._load()

    # ---------- 加载 / 保存 ----------

    def _load(self):
        if INDEX_FILE.exists() and META_FILE.exists():
            # 不用 faiss.read_index：faiss C++ 的 fopen 在含中文的 Windows 路径下会失败，
            # 改用 serialize/deserialize（faiss>=1.15 返回 numpy uint8 数组）+ Python 文件 IO
            self.index = faiss.deserialize_index(np.frombuffer(INDEX_FILE.read_bytes(), dtype=np.uint8))
            with open(META_FILE, "rb") as f:
                meta = pickle.load(f)
            self._texts = meta["texts"]
            self._sources = meta["sources"]
            self._orders = meta.get("orders", list(range(len(self._texts))))
            self._tokenized = meta.get("tokenized")
            # 兼容旧索引（无分词数据）：加载时即时分词构建（一次性，稍慢但无需重建）
            if self._tokenized is None:
                self._tokenized = [_tokenize(t) for t in self._texts]
            if self.index.ntotal != len(self._texts):
                raise RuntimeError(
                    f"索引损坏：faiss 有 {self.index.ntotal} 行，元数据有 {len(self._texts)} 条，"
                    "请删除 faiss_index/ 后重新运行 rebuild_index.py"
                )
        else:
            # 空索引：维度未知（bge-m3 是 1024 维），先不建 faiss 索引，
            # 第一次 add_documents 时按实际嵌入维度创建
            self.index = None
            self._texts, self._sources, self._orders = [], [], []
            self._tokenized = []
        # BM25Okapi 要求至少有一个非空文档，否则内部平均文档长度为 0 会除零
        if any(self._tokenized):
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def save(self):
        """持久化到 faiss_index/ 目录。"""
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("索引为空，无法保存（没有可索引的文档）")
        config.FAISS_INDEX_DIR.mkdir(exist_ok=True)
        # 用 serialize + Python 文件 IO：faiss.write_index 的 fopen 在中文路径下失败
        INDEX_FILE.write_bytes(faiss.serialize_index(self.index).tobytes())
        with open(META_FILE, "wb") as f:
            pickle.dump({
                "texts": self._texts,
                "sources": self._sources,
                "orders": self._orders,
                "tokenized": self._tokenized,
            }, f)

    # ---------- 索引操作 ----------

    def add_documents(self, chunks: list[str], source: str):
        """把一个文档的所有分块向量化并加入索引（一次请求嵌入，避免逐块调用）。"""
        if not chunks:
            return
        embeddings = self.embedder.embed_documents(chunks)
        # L2 归一化：IndexFlatL2 下距离越小越相关，score = 1 - 距离
        matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)

        if self.index is None or self.index.ntotal == 0:
            self.index = faiss.IndexFlatL2(matrix.shape[1])  # 维度以第一次实际嵌入为准

        start = len(self._texts)
        self.index.add(matrix)
        self._texts.extend(chunks)
        self._sources.extend([source] * len(chunks))
        self._orders.extend(range(start, start + len(chunks)))
        # BM25：增量重建语料（rank_bm25 的 fit 是 O(N) 全量，rebuild 场景一次成型，
        # add_documents 仅 rebuild 期间调用，可接受）
        self._tokenized.extend(_tokenize(c) for c in chunks)
        self._bm25 = BM25Okapi(self._tokenized) if any(self._tokenized) else None

    def _bm25_scores(self, query: str) -> list[float]:
        """rank_bm25 BM25Okapi 打分（默认 k1=1.5, b=0.75），返回与索引行一一对应的分数数组。"""
        qt = _tokenize(query)
        if not qt or self._bm25 is None:
            return [0.0] * len(self._texts)
        return list(self._bm25.get_scores(qt))

    def search(self, query: str, top_k: int = None, mode: str | None = None):
        """检索 top_k 个相关块。

        mode：
        - None（默认）：用 config.RETRIEVAL_MODE（hybrid / vector）；
        - "hybrid"：向量召回 top-50 + BM25 召回 top-50，加权 RRF 融合（权重 config.RRF_WEIGHTS）；
        - "vector"：纯向量检索（论文对比实验基线）。

        返回 [{"text", "source", "score"}]，score 为向量相似度（供置信度/MIN_SCORE 使用），
        排序为融合后的相关度排序（vector 模式下为向量相似度排序）。
        """
        top_k = top_k or config.TOP_K
        mode = mode or config.RETRIEVAL_MODE
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)

        # 向量路
        qv = np.array(self.embedder.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(qv)
        qv = qv / max(norm, 1e-12)
        distances, ids = self.index.search(qv.reshape(1, -1), min(VEC_TOP_N, self.index.ntotal))
        vec_ranks: dict[int, int] = {}  # 索引行 -> 排名（0 起）
        for rank, idx in enumerate(ids[0]):
            if idx >= 0:
                vec_ranks[int(idx)] = rank

        if mode == "vector":
            results = []
            for idx in sorted(vec_ranks, key=vec_ranks.get):
                results.append({
                    "text": self._texts[idx],
                    "source": self._sources[idx],
                    "score": 1 - float(distances[0][vec_ranks[idx]]) / 2,
                })
            return results[:k]

        # BM25 路
        bm_scores = self._bm25_scores(query)
        bm_ranks: dict[int, int] = {}
        top_bm = sorted(range(len(bm_scores)), key=lambda i: -bm_scores[i])[:BM25_TOP_N]
        for rank, idx in enumerate(top_bm):
            if bm_scores[idx] > 0:
                bm_ranks[idx] = rank

        # 加权 RRF 融合：score = w_vec/(K+rank_vec+1) + w_bm25/(K+rank_bm25+1)
        # 权重来自 config.RRF_WEIGHTS（论文第 6 章调优实验改这里）
        w_vec = config.RRF_WEIGHTS["vector"]
        w_bm25 = config.RRF_WEIGHTS["bm25"]
        fused: dict[int, float] = {}
        for idx, rank in vec_ranks.items():
            fused[idx] = w_vec / (RRF_K + rank + 1)
        for idx, rank in bm_ranks.items():
            fused[idx] = fused.get(idx, 0.0) + w_bm25 / (RRF_K + rank + 1)

        results = []
        for idx in sorted(fused, key=fused.get, reverse=True)[:k]:
            rank = vec_ranks.get(idx)
            score = 1 - float(distances[0][rank]) / 2 if rank is not None else 0.0
            results.append({
                "text": self._texts[idx],
                "source": self._sources[idx],
                "score": score,
            })
        return results

    # ---------- 重建 ----------

    def rebuild(self, chunk_size: int | None = None, overlap: int | None = None):
        """递归扫描 UPLOAD_DIR 下所有文档，重新解析、分块、建索引（旧索引丢弃重建）。

        - 支持 *.pdf / *.txt / *.docx，子文件夹（主题分类）一并纳入；
        - 元数据 source 记录相对 UPLOAD_DIR 的路径（如 "01_学籍与转专业/xxx.pdf"）；
        - 临时文件跳过，单个文档解析失败/文本为空只打印警告，不中断整体重建。

        chunk_size / overlap：分块参数，None 时用 config 默认值（评测脚本做对比实验时传入）。"""
        # 磁盘文件名是 UUID，展示时用 documents 表里的原始文件名做 source
        from database import SessionLocal
        from models import Document

        db = SessionLocal()
        try:
            # 磁盘文件名是 UUID，展示时用 documents 表里的原始文件名做 source（仅限顶层上传文件）
            name_map = {Path(d.file_path).name: d.filename for d in db.query(Document).all()}
        finally:
            db.close()

        # 索引文件先删除再重建：避免「重建失败但磁盘上还留着旧索引」的假象，
        # 也保证 save() 拿到的是全新结果
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
        if META_FILE.exists():
            META_FILE.unlink()

        docs = list(_iter_upload_documents(config.UPLOAD_DIR))
        if not docs:
            raise RuntimeError(f"UPLOAD_DIR（{config.UPLOAD_DIR}）下没有任何可索引文档（*.pdf/*.txt/*.docx）")

        self.index = None
        self._texts, self._sources, self._orders = [], [], []
        self._tokenized = []
        self._bm25 = None
        for path in docs:
            # source 用相对路径（含主题子文件夹，如 "01_学籍与转专业/xxx.pdf"），
            # 前端上传的 UUID 文件（在 uploads/ 顶层）则换回原始文件名
            rel = path.relative_to(config.UPLOAD_DIR).as_posix()
            source = rel if path.parent != config.UPLOAD_DIR else name_map.get(path.name, rel)
            try:
                text = extract_text(path)
            except ValueError as e:
                print(f"警告：{rel} 解析失败，已跳过（{e}）")
                continue
            if not text.strip():
                print(f"警告：{rel} 提取文本为空，已跳过（可能为扫描版 PDF 或已损坏）")
                continue
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            # 过滤短噪声块（阈值 40 字符，依据实测块长分布）：
            # - OCR 页眉页脚残片、页码、"点击/登录/确认"类操作短句会让余弦分数虚高（短文本偏置），
            #   检索时挤占真实政策条款块的排序位置；
            # - 本知识库实测：政策文档（学籍/转专业/奖学金等）最短块 41+ 字符，
            #   而 CET 报名流程等截图说明文档有大量 13-40 字符的操作短句，
            #   阈值 40 可精准过滤后者而不误伤前者（答辩素材：基于数据分布确定的过滤阈值）
            clean = [c for c in chunks if len(c.strip()) >= 40]
            if len(clean) != len(chunks):
                print(f"过滤短块：{rel}（{len(chunks) - len(clean)} 个 <40 字符的噪声块已丢弃）")
            if not clean:
                print(f"警告：{rel} 过滤后没有有效块，已跳过")
                continue
            self.add_documents(clean, source=source)
            print(f"已索引：{source}（{len(clean)} 个块）")

        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("没有索引到任何块：请检查 UPLOAD_DIR 下是否有可解析的 PDF/TXT/DOCX")
        self.save()
