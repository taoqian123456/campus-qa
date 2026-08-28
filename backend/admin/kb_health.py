"""
知识库体检接口（仅 admin）：文档分块/命中统计 + 僵尸文档检测。

僵尸文档 = 已进入索引但从未被检索命中的文档（hit_count == 0），
提示管理员检查内容质量或删除（答辩素材：基于检索日志的体检机制）。
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from auth.deps import require_admin
from database import get_db
from models import Document, User

router = APIRouter(prefix="/api/admin/kb-health", tags=["管理端-知识库体检"])


@router.get("")
def kb_health(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """知识库体检：每篇文档的分块数/命中次数 + 汇总 + 僵尸文档列表。

    调用前先同步文档注册表与索引分块数（索引可能重建过/首次访问映射为空）。
    """
    from qa.qa_handler import sync_chunk_counts, sync_document_registry

    sync_document_registry()
    sync_chunk_counts()

    docs = db.query(Document).order_by(Document.id).all()
    items = [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "chunk_count": d.chunk_count,
            "hit_count": d.hit_count,
        }
        for d in docs
    ]
    indexed = [d for d in items if d["status"] == "indexed"]
    zombies = [d for d in indexed if d["hit_count"] == 0]
    return {
        "total_documents": len(items),
        "total_chunks": sum(d["chunk_count"] for d in indexed),
        "total_hits": sum(d["hit_count"] for d in indexed),
        "documents": items,
        "zombie_documents": zombies,
    }
