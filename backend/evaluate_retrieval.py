"""
检索命中率对比评测（论文第 6 章实验）：不调 LLM，纯检索质量对比 hybrid vs vector。

用法：
  venv\\Scripts\\python.exe evaluate_retrieval.py                # 两种模式各跑一遍对比
  venv\\Scripts\\python.exe evaluate_retrieval.py --top_k 5 --eval_set eval_set.json

判定标准（自动相关性判据）：
  对每条评测题，把参考答案（reference）jieba 分词去停用词后得到关键词集；
  检索返回的块与关键词集共享 >= HIT_OVERLAP(2) 个关键词视为"相关块"。
  Hit@k = top-k 结果里出现相关块的题目占比；MRR = 首个相关块排名倒数的平均。

说明：这不替代 LLM 层面的相关性/忠实度人工打分（evaluate.py），
但能在不改动 LLM 的情况下客观对比两路检索的召回质量，作为第 6 章实验数据。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jieba

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

STOPWORDS = {
    "的", "了", "是", "吗", "呢", "啊", "吧", "我", "你", "他", "她", "它",
    "在", "有", "和", "与", "及", "或", "就", "都", "也", "还", "要", "想",
    "不", "没", "被", "把", "给", "让", "请", "问", "说", "个", "什么",
    "怎么", "如何", "为什么", "哪些", "哪里", "多少", "可以", "需要",
    "我们", "你们", "他们", "这个", "那个", "一个", "一下", "知道",
    "要点", "按", "须", "应", "经", "由", "等", "前", "后", "内", "上", "下",
}

HIT_OVERLAP = 2  # 共享关键词数达到该值判定为相关块


def keywords_of(text: str) -> set[str]:
    return {w for w in jieba.cut(text) if len(w.strip()) >= 2 and w not in STOPWORDS}


def run_mode(store, items, top_k: int, mode: str) -> dict:
    """跑一种检索模式，返回 {hit1, hit3, hit5, mrr, per_question}。"""
    hits = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    per_question = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        ref_kw = keywords_of(item.get("reference", ""))
        results = store.search(q, top_k, mode=mode)
        first_rel = 0
        for rank, r in enumerate(results, 1):
            chunk_kw = keywords_of(r["text"][:200])  # 取块前 200 字已足够判断主题
            if len(ref_kw & chunk_kw) >= HIT_OVERLAP:
                first_rel = rank
                break
        if first_rel:
            for k in hits:
                if first_rel <= k:
                    hits[k] += 1
            rr_sum += 1.0 / first_rel
        per_question.append((q, first_rel))
    n = len(items)
    return {
        "hit1": hits[1] / n,
        "hit3": hits[3] / n,
        "hit5": hits[5] / n,
        "mrr": rr_sum / n,
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser(description="检索命中率对比：hybrid vs vector（不调 LLM）")
    parser.add_argument("--top_k", type=int, default=5, help="检索条数（默认 5）")
    parser.add_argument("--eval_set", type=str, default="eval_set.json", help="评测集路径")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    eval_path = BASE_DIR / args.eval_set
    items = json.loads(eval_path.read_text(encoding="utf-8"))
    print(f"评测集：{eval_path}（{len(items)} 题） | top_k={args.top_k} | 相关判定：共享关键词>={HIT_OVERLAP}")

    import config
    from qa.vector_store import VectorStore

    print(f"config.RETRIEVAL_MODE={config.RETRIEVAL_MODE} | RRF_WEIGHTS={config.RRF_WEIGHTS}")
    print("=" * 70)

    store = VectorStore()
    results = {}
    for mode in ("hybrid", "vector"):
        t0 = time.perf_counter()
        results[mode] = run_mode(store, items, args.top_k, mode)
        elapsed = time.perf_counter() - t0
        r = results[mode]
        print(f"[{mode:7s}] Hit@1={r['hit1']:.1%}  Hit@3={r['hit3']:.1%}  Hit@5={r['hit5']:.1%}  MRR={r['mrr']:.3f}  ({elapsed:.1f}s)")

    print("=" * 70)
    hy, ve = results["hybrid"], results["vector"]
    # 差异明细：hybrid 赢过的题 / vector 赢过的题 / 平局
    hy_wins = ve_wins = tie = 0
    for (q1, h), (q2, v) in zip(hy["per_question"], ve["per_question"]):
        if h and not v:
            hy_wins += 1
        elif v and not h:
            ve_wins += 1
        else:
            tie += 1
    print(f"题目级对比：hybrid 独赢 {hy_wins} 题 | vector 独赢 {ve_wins} 题 | 同中/同失 {tie} 题")
    print(f"Hit@5 提升：{hy['hit5'] - ve['hit5']:+.1%} | MRR 提升：{hy['mrr'] - ve['mrr']:+.3f}")
    if hy["hit5"] >= ve["hit5"]:
        print("结论：hybrid 混合检索召回质量 ≥ 纯向量，验证了 BM25+向量+RRF 融合的有效性")
    else:
        print("结论：vector 纯向量更优，建议调整 RRF_WEIGHTS 或回退纯向量模式")


if __name__ == "__main__":
    main()
