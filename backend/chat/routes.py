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
from config import LLM_PROVIDER, LLM_PROVIDERS
from database import get_db
from models import ChatSession, Message, User
from qa.qa_handler import answer_question, answer_question_stream, available_llm_providers, recommend_questions

router = APIRouter(prefix="/api/chat", tags=["会话"])


class SessionCreate(BaseModel):
    """新建会话：标题取问题前 20 字。"""
    question: str = Field(..., min_length=1, max_length=2000, description="首条问题")


class MessageCreate(BaseModel):
    """发送消息。history 可选：前端带上当前会话最近几轮对话（[{"question","answer"}]），
    不带时后端从 messages 表查。前端传来的历史会以服务端库中记录为准做校验/补全。
    provider 可选：指定本次回答用的模型厂商（前端顶栏下拉框），空/不传走默认。"""
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] | None = None
    provider: str | None = None


def _resolve_provider(provider: str | None) -> str:
    """校验并解析本次请求使用的模型厂商：空 -> 默认厂商（.env 的 LLM_PROVIDER）；
    不在注册表里 -> 400 并提示可选列表；在注册表但未配置 Key -> 同样 400
    （比等 OpenAI SDK 在流中途抛异常更友好）。返回规范化后的厂商 key。"""
    p = (provider or "").strip()
    if not p:
        p = LLM_PROVIDER
    if p not in LLM_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"不支持的模型 '{p}'，可选：{'、'.join(LLM_PROVIDERS)}")
    if p not in available_llm_providers():
        env = LLM_PROVIDERS[p]["api_key_env"]
        raise HTTPException(status_code=400, detail=f"模型 '{LLM_PROVIDERS[p]['label']}' 未配置，请在 backend/.env 里填 {env}=...")
    return p


class BatchDelete(BaseModel):
    """批量删除会话：ids 为空列表表示清空当前用户全部会话。"""
    ids: list[int] = []


class FeedbackCreate(BaseModel):
    """回答反馈：up 点赞 / down 点踩；down 时建议带原因（预置选项之一）。"""
    message_id: int
    feedback: str = Field(..., pattern="^(up|down)$")
    reason: str | None = None


def _session_out(s: ChatSession) -> dict:
    return {"id": s.id, "title": s.title, "created_at": s.created_at}


def _message_out(m: Message) -> dict:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "provider": m.provider or "deepseek",
        "created_at": m.created_at,
        "feedback": m.reply_feedback,
        "reason": m.reply_reason,
    }


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


@router.get("/models")
def list_models(current_user: User = Depends(get_current_user)):
    """可用模型列表（注册表 -> 前端下拉框选项），登录后可调用。

    只返回"已配置 Key"的模型：用户不会选中一个必然报错的选项；
    返回的 default 若未配置（如 .env 设了 LLM_PROVIDER=kimi 但没填 key），回落到
    已配置列表的第一个。列表内容由服务端统一维护，前端不写死厂商名单。
    """
    models = [{"id": k, "label": v["label"]} for k, v in LLM_PROVIDERS.items()
              if k in available_llm_providers()]
    default = LLM_PROVIDER if any(m["id"] == LLM_PROVIDER for m in models) else (
        models[0]["id"] if models else ""
    )
    return {"models": models, "default": default}


@router.get("/recommend")
def get_recommend(
    question: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """推荐追问：基于检索内容生成 3 个"用户可能接着问"的短问题。

    question 放 query 参数（前端直接 fetch 拼接）；生成失败时 recommend_questions
    内部已静默降级返回 []，这里照常 200——推荐只是增值，不影响主流程。
    """
    return {"questions": recommend_questions(question)}


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


@router.delete("/sessions")
def batch_delete_sessions(
    body: BatchDelete,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """批量删除会话：只删除属于当前用户的会话（ids 为空表示一键清空全部）。

    逐条走归属校验：跨用户/不存在的 id 静默跳过（与单删的 404 不同，
    批量操作以"删掉属于我的那些"为语义，不因个别脏 id 整体失败）。
    """
    ids = body.ids or []
    targets = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id,
        ChatSession.id.in_(ids),
    ).all() if ids else db.query(ChatSession).filter(ChatSession.user_id == current_user.id).all()

    deleted = [s.id for s in targets]
    for s in targets:
        db.query(Message).filter(Message.session_id == s.id).delete()
        db.delete(s)
    db.commit()
    return {"message": "会话已删除", "deleted": deleted}


@router.post("/sessions/{session_id}/messages", status_code=201)
def send_message(
    session_id: int,
    body: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """发送问题：RAG 检索 + 大模型生成回答，问题和答案都写入 messages 表。"""
    session = _get_owned_session(session_id, current_user, db)
    provider = _resolve_provider(body.provider)

    # 历史优先用服务端库中记录（权威、不受前端篡改）；库为空时退回前端传的（无历史会话/刚恢复登录时）
    history = _history_pairs(session_id, db) or (body.history or [])
    result = answer_question(body.question, history=history, provider=provider)

    db.add(Message(session_id=session.id, role="user", content=body.question))
    ai_msg = Message(session_id=session.id, role="assistant", content=result["answer"], provider=provider)
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)  # 取到自增 id，前端点赞需要 message_id
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "provider": provider,
        "message_id": ai_msg.id,
    }


@router.post("/sessions/{session_id}/messages/stream")
def send_message_stream(
    session_id: int,
    body: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """流式问答（SSE）：LLM 用 stream=True 逐块推送，结束后完整答案写入 messages 表。

    provider 校验在流开始前做（非法值直接 400，避免 HTTP 200 + SSE 里夹错误事件的尴尬）。

    事件格式（每行 data: JSON）：
      {"type": "token", "content": "..."}   增量文本
      {"type": "done", "answer": "...", "sources": [...], "provider": "..."}  流结束
      {"type": "error", "message": "..."}   生成失败
    """
    session = _get_owned_session(session_id, current_user, db)
    provider = _resolve_provider(body.provider)
    history = _history_pairs(session_id, db) or (body.history or [])

    def sse_events():
        # 依赖 get_db 的会话在流期间保持存活（同步 def 路由），
        # done 之后才写库并关闭，流提前中断由 get_db 兜底关闭
        yield from _sse_yield(db, session.id, body.question, history, provider)

    return StreamingResponse(
        sse_events(),
        media_type="text/event-stream",
        # 禁止缓冲：X-Accel-Buffering 防 Nginx 等反向代理把 SSE 攒成一大块（症状：前端等不到任何内容）
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_yield(db: Session, session_id: int, question: str, history: list[dict], provider: str):
    """把 answer_question_stream 的字典事件转成 SSE 帧，并在 done 时落库（含模型厂商）。

    落库后把 assistant 消息的自增 id 追加到 done 事件（message_id），前端点赞需要它。
    """
    answer = ""
    done_event = None
    try:
        for event in answer_question_stream(question, history=history, provider=provider):
            if event["type"] == "done":
                answer = event["answer"]
                done_event = dict(event)  # 复制一份，落库后补充 message_id 与 provider
                continue  # done 事件延后发送：先拿到 message_id
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
                ai_msg = Message(session_id=session_id, role="assistant", content=answer, provider=provider)
                db.add(ai_msg)
                db.commit()
                db.refresh(ai_msg)
                if done_event is not None:
                    done_event["message_id"] = ai_msg.id
                    done_event["provider"] = provider
        finally:
            db.close()
    if done_event is not None:
        yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"


@router.post("/feedback")
def submit_feedback(
    body: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """回答反馈：点赞/点踩某条 assistant 消息。

    归属校验：消息必须属于当前用户的某个会话（JOIN sessions），否则 404；
    只允许对 assistant 消息打反馈，user 消息返回 400。
    """
    msg = (
        db.query(Message)
        .join(ChatSession, ChatSession.id == Message.session_id)
        .filter(Message.id == body.message_id, ChatSession.user_id == current_user.id)
        .first()
    )
    if msg is None:
        raise HTTPException(status_code=404, detail="消息不存在")
    if msg.role != "assistant":
        raise HTTPException(status_code=400, detail="只能对 AI 回答进行反馈")

    msg.reply_feedback = body.feedback
    # down 时保留原因；up 时清掉旧原因
    msg.reply_reason = body.reason if body.feedback == "down" else None
    db.commit()
    return {"message": "反馈已记录", "message_id": msg.id, "feedback": msg.reply_feedback, "reason": msg.reply_reason}


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
