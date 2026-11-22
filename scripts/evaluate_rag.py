"""Đánh giá phần RAG trên bộ câu hỏi tests/rag_eval_questions.yml.

    python scripts/evaluate_rag.py              # chỉ đánh giá tìm kiếm (không cần Gemini)
    python scripts/evaluate_rag.py --answer     # đánh giá cả câu trả lời của Gemini (cần GEMINI_API_KEY)

Kết quả ghi vào results/rag_eval.csv và results/rag_eval.md.
- Retrieval hit@5: tất cả chuỗi expected nằm trong 5 đoạn tài liệu sau rerank.
- Answer accuracy: tất cả chuỗi expected nằm trong câu trả lời của Gemini.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from actions import action_rag_retrieval as rag  # noqa: E402  (nạp model khi import)


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def contains_all(text: str, expected) -> bool:
    t = norm(text)
    return all(norm(e) in t for e in expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", action="store_true", help="gọi Gemini để chấm câu trả lời")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "tests" / "rag_eval_questions.yml")
    args = parser.parse_args()

    questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))["questions"]
    collection = rag._load_collection()
    if collection is None or collection.count() == 0:
        sys.exit("Chưa có dữ liệu trong ChromaDB. Chạy: python scripts/ingest_documents.py")

    action = rag.ActionRAGRetrieval()
    rows = []
    for i, q in enumerate(questions, 1):
        t0 = time.time()
        docs = action._search_chromadb(collection, q["question"], top_k=10)
        docs = action._rerank_documents(q["question"], docs, top_k=5)
        t_retrieval = time.time() - t0
        context = action._build_context(docs)
        hit = contains_all(context, q["expected"])

        answer, correct, t_answer = "", None, 0.0
        if args.answer:
            t1 = time.time()
            answer = action._generate_answer(q["question"], context, "Chưa có lịch sử trò chuyện.")
            t_answer = time.time() - t1
            correct = contains_all(answer, q["expected"])

        rows.append({
            "question": q["question"],
            "expected": "; ".join(q["expected"]),
            "retrieval_hit": hit,
            "answer_correct": correct,
            "sources": "; ".join(sorted({d.get("source") or "" for d in docs})),
            "retrieval_s": round(t_retrieval, 2),
            "answer_s": round(t_answer, 2),
            "answer": answer.replace("\n", " "),
        })
        mark = "OK " if hit else "MISS"
        extra = "" if correct is None else (" | trả lời ĐÚNG" if correct else " | trả lời SAI")
        print(f"[{i:02d}] {mark} {q['question']}{extra}")

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "rag_eval.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    hits = sum(r["retrieval_hit"] for r in rows)
    lines = [
        "# Kết quả đánh giá RAG",
        "",
        f"- Số câu hỏi: {n}",
        f"- Retrieval hit@5: {hits}/{n} = {hits / n:.1%}",
        f"- Thời gian tìm kiếm trung bình: {sum(r['retrieval_s'] for r in rows) / n:.2f} s",
    ]
    if args.answer:
        ok = sum(bool(r["answer_correct"]) for r in rows)
        lines += [
            f"- Độ chính xác câu trả lời: {ok}/{n} = {ok / n:.1%}",
            f"- Thời gian sinh câu trả lời trung bình: {sum(r['answer_s'] for r in rows) / n:.2f} s",
        ]
    lines += ["", "| # | Câu hỏi | Kỳ vọng | Tìm đúng | Trả lời đúng |", "|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        ans = "-" if r["answer_correct"] is None else ("✔" if r["answer_correct"] else "✘")
        lines.append(f"| {i} | {r['question']} | {r['expected']} | {'✔' if r['retrieval_hit'] else '✘'} | {ans} |")
    (out_dir / "rag_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7]))
    print(f"Chi tiết: {out_dir / 'rag_eval.md'}")


if __name__ == "__main__":
    main()
