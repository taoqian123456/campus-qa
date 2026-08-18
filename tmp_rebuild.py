import sys
sys.path.insert(0, r"D:\deepseek\创作1\campus-qa\backend")
from qa.vector_store import VectorStore
store = VectorStore()
store.rebuild()
print("REBUILD_OK total=", store.index.ntotal)
