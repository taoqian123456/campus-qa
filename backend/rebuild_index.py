"""
重建知识库向量索引（可单独运行）。

流程：递归扫描 UPLOAD_DIR（含主题子文件夹）-> extract_text -> chunk_text -> 嵌入 -> FAISS 索引 -> save。
支持 *.pdf、*.txt、*.docx；元数据 source 记录相对路径（如 "01_学籍与转专业/xxx.pdf"）。
用法：venv\\Scripts\\python.exe rebuild_index.py
"""
import sys
from pathlib import Path

# 脚本放在 backend/ 根目录，保证任何目录下执行都能 import 项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qa.vector_store import VectorStore


def main():
    store = VectorStore()
    store.rebuild()
    print(f"共索引 {store.index.ntotal} 个块")


if __name__ == "__main__":
    main()
