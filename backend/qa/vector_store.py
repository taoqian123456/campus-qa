"""
FAISS 向量索引：把知识库分块向量化、持久化到 faiss_index/ 目录，并支持语义检索。

存储结构：
  faiss_index/index.faiss   向量（IndexFlatL2，向量已做 L2 归一化，平方距离/2 = 1 - 余弦相似度）
  faiss_index/index.pkl     元数据：块文本、来源文件名、块在文档中的顺序
"""
import pickle
from pathlib import Path

import faiss
import numpy as np

from config import FAISS_INDEX_DIR, TOP_K
from qa.embeddings import get_embedder
from qa.knowledge_base import chunk_text, extract_text

INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
META_FILE = FAISS_INDEX_DIR / "index.pkl"

# 扫描时跳过的临时/系统文件；~$ 开头是 Office 锁文件，._ 开头是 macOS AppleDouble
TEMP_FILENAMES = {".ds_store", "thumbs.db", "desktop.ini"}
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx"}  # 与 admin.documents.ALLOWED_SUFFIXES 保持一致


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
      faiss 第 i 行  <->  self._texts[i] / self._sources[i]
    顺序（order）留给未来「按文档内顺序拼接上下文」用，暂未使用。
    """

    def __init__(self, embedder=None):
        """加载已有索引；不存在则建空索引。"""
        self.embedder = embedder or get_embedder()
        self.index: faiss.Index | None = None
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._orders: list[int] = []
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

    def save(self):
        """持久化到 faiss_index/ 目录。"""
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("索引为空，无法保存（没有可索引的文档）")
        FAISS_INDEX_DIR.mkdir(exist_ok=True)
        # 用 serialize + Python 文件 IO：faiss.write_index 的 fopen 在中文路径下失败
        INDEX_FILE.write_bytes(faiss.serialize_index(self.index).tobytes())
        with open(META_FILE, "wb") as f:
            pickle.dump({"texts": self._texts, "sources": self._sources, "orders": self._orders}, f)

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

    def search(self, query: str, top_k: int = TOP_K):
        """语义检索：返回 [{"text", "source", "score"}]，按相关度降序。"""
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        qv = np.array(self.embedder.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(qv)
        qv = qv / max(norm, 1e-12)
        distances, ids = self.index.search(qv.reshape(1, -1), k)
        results = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx < 0:
                continue
            # 向量均归一化；faiss 返回平方 L2 距离，cos = 1 - dist/2，落在 [-1, 1]，越大越相关
            score = 1 - float(dist) / 2
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
        from config import UPLOAD_DIR

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

        docs = list(_iter_upload_documents(UPLOAD_DIR))
        if not docs:
            raise RuntimeError(f"UPLOAD_DIR（{UPLOAD_DIR}）下没有任何可索引文档（*.pdf/*.txt/*.docx）")

        self.index = None
        self._texts, self._sources, self._orders = [], [], []
        for path in docs:
            # source 用相对路径（含主题子文件夹，如 "01_学籍与转专业/xxx.pdf"），
            # 前端上传的 UUID 文件（在 uploads/ 顶层）则换回原始文件名
            rel = path.relative_to(UPLOAD_DIR).as_posix()
            source = rel if path.parent != UPLOAD_DIR else name_map.get(path.name, rel)
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
