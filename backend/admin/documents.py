"""
文档管理接口（仅 admin）：上传、列表、删除。

权限说明：文档表里没有上传者字段，任何 admin 都可以查看/删除全部文档，
这是毕设规模的合理简化，符合原始表结构设计。
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth.deps import require_admin
from config import UPLOAD_DIR
from database import get_db
from models import Document, User

router = APIRouter(prefix="/api/admin/documents", tags=["管理端-知识库"])

# 允许上传的格式（P3 建索引时用 qa.knowledge_base.extract_text 解析）
ALLOWED_SUFFIXES = {".pdf", ".txt", ".docx"}


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """上传知识库文档：保存到 UPLOAD_DIR 并写入 documents 表（status=pending）。

    这一步只保存文件，不做文本解析——解析留到 P3 构建索引时进行。
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持 {'/'.join(sorted(ALLOWED_SUFFIXES))} 格式，收到: {suffix or '(无扩展名)'}",
        )

    # 随机文件名，避免中文/重复文件名带来的路径问题
    saved_name = f"{uuid.uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name

    content = await file.read()
    saved_path.write_bytes(content)

    doc = Document(
        filename=file.filename,
        file_path=str(saved_path),
        status="pending",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {
        "id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "created_at": doc.created_at,
    }


@router.get("")
def list_documents(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """列出全部文档（按 id 倒序）。"""
    docs = db.scalars(select(Document).order_by(Document.id.desc())).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "created_at": d.created_at,
        }
        for d in docs
    ]


@router.delete("/{doc_id}")
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """删除文档：数据库记录 + 磁盘文件（磁盘文件已不存在也照常删记录）。"""
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 只允许删除 UPLOAD_DIR 内的文件，防止误删别处文件
    file_path = Path(doc.file_path)
    if file_path.parent == UPLOAD_DIR and file_path.exists():
        file_path.unlink()

    db.delete(doc)
    db.commit()
    return {"message": "文档已删除", "id": doc_id, "filename": doc.filename}


@router.post("/rebuild")
def rebuild_index(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """重建向量索引：扫描 uploads/ 全部文档，重建完成后把 documents 表状态同步为 indexed。

    注意：上传记录被删除但磁盘文件还在的文档，索引重建后会把 status 置为 indexed
    （status 表示"已进入索引"，不要求有对应上传记录）；失败时整体返回 500。
    """
    from qa.vector_store import VectorStore

    store = VectorStore()
    try:
        store.rebuild()
        total = store.index.ntotal
    except Exception as e:
        # 已删除旧索引文件、但重建中途失败：此时没有可用索引，
        # 问答会走「知识库为空」的兜底回复，向管理员如实报告
        raise HTTPException(status_code=500, detail=f"重建索引失败：{e}")

    db.query(Document).update({"status": "indexed"})
    db.commit()
    return {"message": "索引重建成功", "total_chunks": total}
