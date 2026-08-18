"""
RAG 性能测试脚本（P9）：20 个真实问题，统计单次与平均耗时。

用法：
  venv\\Scripts\\python.exe perf_test.py
  venv\\Scripts\\python.exe perf_test.py --n 20 --output perf_results.csv

说明：
- 问题从评测集 eval_set.json 取前 N 条（没有评测集则用内置默认问题列表）；
- 直接调 qa_handler.answer_question（与评测一致，不经过 HTTP）；
- 输出：控制台逐条耗时 + CSV（question, elapsed_s, answer）。
"""
import argparse
import csv
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

DEFAULT_QUESTIONS = [
    "转专业需要满足什么条件？",
    "转专业申请在什么时间？",
    "哪些学生不能申请转专业？",
    "一等奖学金的条件是什么？",
    "奖学金可以重复享受吗？",
    "哪些课程必须重修？",
    "重修申请什么时候办理？",
    "授予学士学位对绩点有什么要求？",
    "GPA 低于 2.0 还能拿到学位吗？",
    "学位证书丢失了怎么办？",
    "新生不能按期报到怎么办？",
    "保留入学资格的条件是什么？",
    "学分认定和转换什么时候办理？",
    "什么情况下可以认定职业资格类学分？",
    "CET 四六级报名网站是什么？",
    "惠州校区怎么借书？",
    "学业预警有哪些类别？",
    "优秀毕业生评选条件是什么？",
    "毕业证可以代领吗？",
    "什么情况下证书会被撤销？",
]


def load_questions(n: int) -> list[str]:
    """优先从评测集取前 N 条问题。"""
    eval_path = BASE_DIR / "eval_set.json"
    if eval_path.exists():
        import json

        with open(eval_path, encoding="utf-8") as f:
            items = json.load(f)
        if items:
            return [str(x["question"]) for x in items[:n]]
    return DEFAULT_QUESTIONS[:n]


def main():
    parser = argparse.ArgumentParser(description="RAG 性能测试：统计问答耗时")
    parser.add_argument("--n", type=int, default=20, help="测试问题数量（默认 20）")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径（默认 perf_results.csv）")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    questions = load_questions(args.n)
    print(f"性能测试：{len(questions)} 个问题")
    print("=" * 70)

    from qa.qa_handler import answer_question

    rows = []
    times = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            result = answer_question(q)
            answer = result.get("answer", "")
        except Exception as e:
            answer = f"[出错] {e}"
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        rows.append({"question": q, "elapsed_s": round(elapsed, 2), "answer": answer})
        print(f"[{i}/{len(questions)}] {elapsed:5.1f}s  {q}")

    avg = sum(times) / len(times) if times else 0
    total = sum(times)
    print("=" * 70)
    print(f"平均耗时：{avg:.1f}s | 总耗时：{total:.1f}s | 最慢：{max(times):.1f}s | 最快：{min(times):.1f}s")

    out_path = Path(args.output) if args.output else BASE_DIR / "perf_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "耗时(秒)", "answer"])
        for r in rows:
            writer.writerow([r["question"], r["elapsed_s"], r["answer"]])
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
