"""
RAG 评测脚本（P8）：读取评测集，逐题问答，生成打分 CSV。

用法：
  venv\\Scripts\\python.exe evaluate.py                          # 用当前 config 参数，不重建索引
  venv\\Scripts\\python.exe evaluate.py --chunk_size 400 --top_k 5
  venv\\Scripts\\python.exe evaluate.py --chunk_size 200 --top_k 5 --eval_set eval_set.json --output out.csv

说明：
  --chunk_size 变化会重建索引（分块方式变了，不重建没有意义）；重建会覆盖磁盘上的 faiss_index/，
  跑完对比实验后记得用管理后台或 rebuild_index.py 恢复正式参数。
  --top_k 只影响检索条数，不触发重建。

输出 CSV 列：question, answer, reference, 相关性评分(1-5), 忠实度评分(1-5)
（两个评分列留空，在 Excel 里人工打分；UTF-8 BOM 编码，Excel 直接打开不乱码。）
"""
import argparse
import csv
import sys
import time
from pathlib import Path

# 脚本放在 backend/ 根目录，保证任何目录下执行都能 import 项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_DIR = Path(__file__).resolve().parent


def apply_overrides(chunk_size: int | None, top_k: int | None):
    """把命令行参数覆盖到 config 及所有已绑定引用处，返回覆盖说明。

    注意：qa_handler 里是 from config import TOP_K，模块属性在导入时绑定，
    必须连同 qa_handler.TOP_K 一起改；chunk_size 通过 rebuild(chunk_size=...) 显式传入，无需改绑定。
    """
    import config
    import qa.qa_handler as qh

    notes = []
    if top_k is not None:
        config.TOP_K = top_k
        qh.TOP_K = top_k  # 同步已绑定的模块属性
        notes.append(f"top_k={top_k}")
    if chunk_size is not None:
        config.CHUNK_SIZE = chunk_size
        notes.append(f"chunk_size={chunk_size}")
    return ", ".join(notes) or "默认参数"


def load_eval_set(path: Path) -> list[dict]:
    import json

    if not path.exists():
        raise FileNotFoundError(
            f"评测集不存在：{path}\n"
            f"参考格式见 prompts/eval_set.example.json，可复制为 backend/{path.name} 后填入你的题目"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"评测集格式错误：应为非空数组 [{{'question': ..., 'reference': ...}}, ...]")
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict) or "question" not in item or "reference" not in item:
            raise ValueError(f"第 {i} 条缺少 question 或 reference 字段：{item}")
    return data


def main():
    parser = argparse.ArgumentParser(description="RAG 评测：逐题问答并生成打分 CSV")
    parser.add_argument("--chunk_size", type=int, default=None, help="覆盖 config.CHUNK_SIZE（触发重建索引）")
    parser.add_argument("--top_k", type=int, default=None, help="覆盖 config.TOP_K（不重建索引）")
    parser.add_argument("--eval_set", type=str, default="eval_set.json", help="评测集路径（默认 backend/eval_set.json）")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径（默认 eval_results_c{chunk}_k{top}.csv）")
    parser.add_argument("--no_rebuild", action="store_true", help="即使指定了 --chunk_size 也不重建索引（已有对应索引时用）")
    args = parser.parse_args()

    # Windows 控制台默认 GBK，打印中文可能报错，统一转 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    eval_set_path = BASE_DIR / args.eval_set
    items = load_eval_set(eval_set_path)
    notes = apply_overrides(args.chunk_size, args.top_k)
    print(f"评测集：{eval_set_path}（{len(items)} 题）")
    print(f"参数：{notes}")
    print("=" * 70)

    # 分块参数变化 -> 重建索引（新分块方式下重建，旧索引作废）
    if args.chunk_size is not None and not args.no_rebuild:
        from qa.vector_store import VectorStore

        print(f"[重建索引] chunk_size={args.chunk_size} ...")
        store = VectorStore()
        store.rebuild(chunk_size=args.chunk_size)
        print(f"[重建索引] 完成，共 {store.index.ntotal} 个块")
        print("=" * 70)

    # 逐题问答（直接调 qa_handler.answer_question，避免 HTTP 与流式开销）
    from qa.qa_handler import answer_question

    rows = []
    for i, item in enumerate(items, 1):
        question = item["question"]
        reference = item.get("reference", "")
        print(f"\n[{i}/{len(items)}] 问题：{question}")
        print(f"参考要点：{reference}")
        t0 = time.perf_counter()
        try:
            result = answer_question(question)
            answer = result["answer"]
            sources = result.get("sources", [])
            print(f"回答：{answer}")
            if sources:
                print(f"来源：{'、'.join(sources)}")
        except Exception as e:
            answer = f"[评测出错] {e}"
            print(answer)
        elapsed = time.perf_counter() - t0
        print(f"耗时：{elapsed:.1f}s")
        rows.append({
            "question": question,
            "answer": answer,
            "reference": reference,
            "relevance": "",   # 人工打分
            "faithfulness": "",  # 人工打分
        })

    # 输出 CSV（utf-8-sig 带 BOM，Excel 直接打开不乱码）
    c = args.chunk_size or "default"
    k = args.top_k or "default"
    out_path = Path(args.output) if args.output else BASE_DIR / f"eval_results_c{c}_k{k}.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "reference", "相关性评分(1-5)", "忠实度评分(1-5)"])
        for r in rows:
            writer.writerow([r["question"], r["answer"], r["reference"], "", ""])

    print("\n" + "=" * 70)
    print(f"完成：{len(rows)} 题已写入 {out_path}")
    print("在 Excel 里填相关性/忠实度评分（1-5），多组参数跑完后比较平均分 → 论文第 6 章表格")


if __name__ == "__main__":
    main()
