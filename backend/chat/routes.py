"""
会话与消息接口（登录用户）。
"""
import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from database import get_db
from models import ChatSession, Message, User
from qa.qa_handler import answer_question, answer_question_stream

router = APIRouter(prefix="/api/chat", tags=["会话"])


class SessionCreate(BaseModel):
    """新建会话：标题取问题前 20 字。"""
    question: str = Field(..., min_length=1, max_length=2000, description="首条问题")


class MessageCreate(BaseModel):
    """发送消息。"""
    question: str = Field(..., min_length=1, max_length=2000)


def _session_out(s: ChatSession) -> dict:
    return {"id": s.id, "title": s.title, "created_at": s.created_at}


def _message_out(m: Message) -> dict:
    return {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at}


def _get_owned_session(session_id: int, user: User, db: Session) -> ChatSession:
    """取会话并校验归属：不存在或不属于当前用户都返回 404（不暴露会话存在性）。"""
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


def _history_pairs(session_id: int, db: Session) -> list[dict]:
    """把会话历史拼成 [{"question", "answer"}]（user/assistant 相邻配对）。"""
    msgs = db.scalars(
        select(Message).where(Message.session_id == session_id).order_by(Message.id)
    ).all()
    pairs = []
    for m in msgs:
        if m.role == "user":
            pairs.append({"question": m.content, "answer": ""})
        elif m.role == "assistant" and pairs:
            pairs[-1]["answer"] = m.content
    return pairs


@router.post("/sessions", status_code=201)
def create_session(
    body: SessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """新建会话：标题取问题前 20 字。"""
    title = body.question[:20] + ("..." if len(body.question) > 20 else "")
    session = ChatSession(user_id=current_user.id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return _session_out(session)


@router.get("/sessions")
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """当前用户的会话列表（按创建时间倒序）。"""
    sessions = db.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
    ).all()
    return [_session_out(s) for s in sessions]


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除会话（同时删除该会话下的全部消息）。"""
    session = _get_owned_session(session_id, current_user, db)
    db.query(Message).filter(Message.session_id == session.id).delete()
    db.delete(session)
    db.commit()
    return {"message": "会话已删除", "id": session_id}


@router.post("/sessions/{session_id}/messages", status_code=201)
def send_message(
    session_id: int,
    body: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """发送问题：RAG 检索 + DeepSeek 生成回答，问题和答案都写入 messages 表。"""
    session = _get_owned_session(session_id, current_user, db)

    # 带最近几轮历史（qa_handler 内部最多取 3 轮）
    history = _history_pairs(session_id, db)
    result = answer_question(body.question, history=history)

    db.add(Message(session_id=session.id, role="user", content=body.question))
    db.add(Message(session_id=session.id, role="assistant", content=result["answer"]))
    db.commit()
    return {"answer": result["answer"], "sources": result["sources"]}


@router.post("/sessions/{session_id}/messages/stream")
def send_message_stream(
    session_id: int,
    body: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """流式问答（SSE）：LLM 用 stream=True 逐块推送，结束后完整答案写入 messages 表。

    事件格式（每行 data: JSON）：
      {"type": "token", "content": "..."}   增量文本
      {"type": "done", "answer": "...", "sources": [...]}  流结束
      {"type": "error", "message": "..."}   生成失败
    """
    session = _get_owned_session(session_id, current_user, db)
    history = _history_pairs(session_id, db)

    def sse_events():
        # 依赖 get_db 的会话在流期间保持存活（同步 def 路由），
        # done 之后才写库并关闭，流提前中断由 get_db 兜底关闭
        yield from _sse_yield(db, session.id, body.question, history)

    return StreamingResponse(
        sse_events(),
        media_type="text/event-stream",
        # 禁止缓冲：X-Accel-Buffering 防 Nginx 等反向代理把 SSE 攒成一大块（症状：前端等不到任何内容）
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_yield(db: Session, session_id: int, question: str, history: list[dict]):
    """把 answer_question_stream 的字典事件转成 SSE 帧，并在 done 时落库。"""
    answer = ""
    try:
        for event in answer_question_stream(question, history=history):
            if event["type"] == "done":
                answer = event["answer"]
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as e:
        # 检索/嵌入等意外异常：转成 error 事件，避免流直接断掉
        yield f"data: {json.dumps({'type': 'error', 'message': f'生成回答失败：{e}'}, ensure_ascii=False)}\n\n"
    finally:
        # 无论正常结束、异常还是客户端断开，都把已生成的内容落库
        # （异常时 answer 可能为空，空答案不写库，避免污染历史）
        try:
            if answer:
                db.add(Message(session_id=session_id, role="user", content=question))
                db.add(Message(session_id=session_id, role="assistant", content=answer))
                db.commit()
        finally:
            db.close()


@router.get("/sessions/{session_id}/messages")
def list_messages(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """会话历史消息（按时间正序）。"""
    _get_owned_session(session_id, current_user, db)
    msgs = db.scalars(
        select(Message).where(Message.session_id == session_id).order_by(Message.id)
    ).all()
    return [_message_out(m) for m in msgs]
