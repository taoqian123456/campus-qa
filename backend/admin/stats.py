"""
统计接口（仅 admin）：热词统计、回答反馈统计。

- 热词：从 messages 表取最近 7 天的 user 消息，jieba 分词后统计 Top 20 高频词；
- 反馈：messages.reply_feedback 的 up/down 总数 + down 原因分布（reply_reason）。
"""
from datetime import datetime, timedelta

import jieba
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from admin.documents import require_admin
from database import get_db
from models import Message, User

router = APIRouter(prefix="/api/admin/stats", tags=["管理端-统计"])

STOPWORDS = {
    "的", "了", "是", "吗", "呢", "啊", "吧", "我", "你", "他", "她", "它",
    "在", "有", "和", "与", "及", "或", "就", "都", "也", "还", "要", "想",
    "不", "没", "被", "把", "给", "让", "请", "问", "说", "个", "什么",
    "怎么", "如何", "为什么", "哪些", "哪里", "多少", "可以", "需要",
    "我们", "你们", "他们", "这个", "那个", "一个", "一下", "知道",
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "for", "in", "on",
}

# 有意义的词至少 2 个字符（过滤单字虚词和切剩的标点）
MIN_WORD_LEN = 2


@router.get("/hotwords")
def hotwords(
    days: int = 7,
    top: int = 20,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """最近 days 天用户提问的高频词 Top top（默认 7 天 / 20 词）。"""
    since = datetime.now() - timedelta(days=days)
    rows = db.scalars(
        select(Message.content).where(
            Message.role == "user",
            Message.created_at >= since,
        )
    ).all()
    if not rows:
        return {"days": days, "total_messages": 0, "words": []}

    freq: dict[str, int] = {}
    for text in rows:
        for w in jieba.cut(text):
            w = w.strip()
            if (
                len(w) >= MIN_WORD_LEN
                and w not in STOPWORDS
                and not w.isascii()   # 过滤纯英文/数字碎片（毕设问答以中文为主）
            ):
                freq[w] = freq.get(w, 0) + 1

    top_words = [
        {"word": w, "count": c}
        for w, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    ]
    return {"days": days, "total_messages": len(rows), "words": top_words}


@router.get("/feedback")
def feedback_stats(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """回答反馈统计：up/down 总数 + down 原因分布（按数量倒序）。"""
    counts = dict(
        db.execute(
            select(Message.reply_feedback, func.count())
            .where(Message.reply_feedback.isnot(None))
            .group_by(Message.reply_feedback)
        ).all()
    )
    reasons = db.execute(
        select(Message.reply_reason, func.count())
        .where(Message.reply_feedback == "down", Message.reply_reason.isnot(None))
        .group_by(Message.reply_reason)
    ).all()
    return {
        "up": counts.get("up", 0),
        "down": counts.get("down", 0),
        "reasons": [
            {"reason": r, "count": c}
            for r, c in sorted(reasons, key=lambda kv: -kv[1])
        ],
    }
