"""Nạp tài liệu tuyển sinh vào ChromaDB.

Đặt tài liệu (.pdf, .docx, .txt, .md) vào docs/admission/ rồi chạy từ thư mục gốc:

    python scripts/ingest_documents.py

Script sẽ xoá và tạo lại collection (mặc định: soict_admission), các collection khác giữ nguyên.
PDF dạng ảnh scan (không bôi đen được chữ) cần OCR trước, hoặc chuyển nội dung sang .txt/.md.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs" / "admission"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "soict_admission")
EMBEDDING_MODEL = "BAAI/bge-m3"  # phải trùng với actions/action_rag_retrieval.py
CHUNK_SIZE = 800  # ký tự
CHUNK_OVERLAP = 150
BATCH_SIZE = 16
SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def clean_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"([•▪◦])\s*\n\s*", r"\1 ", text)  # gạch đầu dòng PDF bị tách dòng
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_page(page) -> str:
    """Đọc 1 trang PDF: bảng giữ nguyên theo hàng ("a | b | c"), bỏ chân trang."""
    try:
        tables = list(page.find_tables())
    except Exception:  # pymupdf cũ không có find_tables
        tables = []
    rects = [fitz_rect(t.bbox) for t in tables]
    words = page.get_text("words") if tables else []
    footer_y = page.rect.height - 45  # vùng chân trang (tên tài liệu, số trang)

    parts = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if y0 > footer_y or not text.strip():
            continue
        if any(r.contains(fitz_rect((x0, y0, x1, y1))) for r in rects):
            continue
        parts.append((y0, text.strip()))
    for t, r in zip(tables, rects):
        # Lấy chữ từng ô theo toạ độ từ (t.extract() hay dính chữ, lệch dấu thập phân)
        grid = []
        for row in t.rows:
            cells = [cell_text(words, c) if c else "" for c in row.cells]
            cells = [c for c in cells if c]
            if cells:
                grid.append(cells)
        if not grid:
            continue
        header, lines = grid[0], [" | ".join(grid[0])]
        # Hàng đầu chỉ là tiêu đề nếu không chứa ô toàn số (bảng bị ngắt trang thì hàng đầu là dữ liệu)
        has_header = not any(re.fullmatch(r"[\d.,]+", c) for c in header)
        for cells in grid[1:]:
            if has_header and len(cells) == len(header) and cells != header:
                # "Cột: giá trị" giúp mỗi dòng tự mang ngữ cảnh khi bị cắt thành đoạn riêng
                lines.append("; ".join(f"{h}: {v}" for h, v in zip(header, cells)))
            elif cells != header:
                lines.append(" | ".join(cells))
        parts.append((r.y0, "\n".join(lines)))
    parts.sort(key=lambda p: p[0])
    return "\n\n".join(text for _, text in parts)


def cell_text(words, bbox) -> str:
    """Ghép các từ có tâm nằm trong ô bảng, theo thứ tự dòng rồi cột."""
    x0, y0, x1, y1 = bbox
    inside = [
        w for w in words
        if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1
    ]
    inside.sort(key=lambda w: (w[5], w[6], w[7]))  # block, line, word
    return " ".join(w[4] for w in inside).strip()


def fitz_rect(bbox):
    import fitz

    return fitz.Rect(bbox) + (-1, -1, 1, 1)  # nới 1pt để so sánh vùng chứa


def read_pages(path: Path) -> List[Dict]:
    """Trả về danh sách {'text', 'page'}; page=None với file không phân trang."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz  # pymupdf

        with fitz.open(path) as pdf:
            title = (pdf.metadata or {}).get("title") or ""
            pages = [{"text": read_pdf_page(page), "page": i + 1} for i, page in enumerate(pdf)]
        if title and pages:
            pages[0]["text"] = f"# {title}\n\n{pages[0]['text']}"
        return pages
    if suffix == ".docx":
        import docx  # python-docx

        document = docx.Document(str(path))
        lines = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                lines.append(" | ".join(cell.text.strip() for cell in row.cells))
        return [{"text": "\n".join(lines), "page": None}]
    return [{"text": path.read_text(encoding="utf-8"), "page": None}]


def split_text(text: str) -> List[str]:
    """Cắt theo đoạn văn, gộp đến ~CHUNK_SIZE ký tự, có chồng lấn để giữ ngữ cảnh."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > CHUNK_SIZE:  # đoạn quá dài: cắt cứng
            head, para = para[:CHUNK_SIZE], para[CHUNK_SIZE - CHUNK_OVERLAP :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-CHUNK_OVERLAP:] if current else ""
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        chunks.append(current)
    return chunks


def build_chunks(docs_dir: Path) -> List[Dict]:
    files = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"Không có tài liệu (.pdf/.docx/.txt/.md) trong {docs_dir}")

    items: List[Dict] = []
    for path in files:
        n_before = len(items)
        pages = read_pages(path)
        # Tiêu đề tài liệu (dòng "# ..." đầu tiên hoặc tên file) được gắn vào mỗi đoạn
        # để đoạn cắt rời vẫn biết nó nói về chương trình / năm nào.
        first = pages[0]["text"].lstrip().split("\n", 1)[0] if pages else ""
        title = first[2:].strip() if first.startswith("# ") else path.stem
        for page in pages:
            for chunk in split_text(clean_text(page["text"])):
                if len(chunk) < 30:
                    continue
                if not chunk.startswith(f"# {title}"):
                    chunk = f"[{title}]\n{chunk}"
                meta = {"source": path.name}
                if page["page"] is not None:
                    meta["page"] = page["page"]
                uid = hashlib.md5(f"{path.name}|{page['page']}|{chunk}".encode()).hexdigest()
                items.append({"id": uid, "text": chunk, "metadata": meta})
        print(f"  {path.name}: {len(items) - n_before} đoạn")
        if len(items) == n_before and path.suffix.lower() == ".pdf":
            print("    ! Không đọc được chữ - có thể là PDF scan, cần OCR.")

    # bỏ đoạn trùng lặp
    unique = {item["id"]: item for item in items}
    return list(unique.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR)
    args = parser.parse_args()

    print(f"Đọc tài liệu trong {args.docs}")
    items = build_chunks(args.docs)
    print(f"Tổng cộng {len(items)} đoạn.")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer(EMBEDDING_MODEL)
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        texts = [it["text"] for it in batch]
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        collection.add(
            ids=[it["id"] for it in batch],
            documents=texts,
            metadatas=[it["metadata"] for it in batch],
            embeddings=embeddings,
        )
        print(f"Đã nạp {start + len(batch)}/{len(items)}")

    print(f"Xong. Collection '{COLLECTION_NAME}' có {collection.count()} đoạn.")


if __name__ == "__main__":
    main()
