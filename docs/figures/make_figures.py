"""Vẽ các sơ đồ minh hoạ cho báo cáo: python docs/figures/make_figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "Arial"

RED, DARK, BLUE, GREEN, GREY, ORANGE = "#C8102E", "#7A0010", "#1F5FAD", "#2E7D32", "#555555", "#E07B00"


def box(ax, x, y, w, h, title, sub="", color=BLUE, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=color, lw=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.32), w, 0.32, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=color, lw=2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="center", color="white", fontsize=fs, weight="bold")
    if sub:
        ax.text(x + w / 2, y + (h - 0.32) / 2, sub, ha="center", va="center", fontsize=fs - 2, color="#222",
                linespacing=1.35)


def arrow(ax, p1, p2, text="", color=GREY, rad=0.0, fs=9, offset=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=1.8, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        mx, my = (p1[0] + p2[0]) / 2 + offset[0], (p1[1] + p2[1]) / 2 + offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=color,
                bbox=dict(fc="white", ec="none", pad=1.5))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ---------------- 1. Kiến trúc tổng thể ----------------
fig, ax = canvas(16, 6.0)
box(ax, 0.2, 2.2, 2.2, 1.6, "Người dùng", "Thí sinh / phụ huynh\n(trình duyệt web)", GREY)
box(ax, 3.2, 2.2, 2.4, 1.6, "Streamlit UI", "streamlit_app.py\ncổng 8501", RED)
box(ax, 6.4, 1.8, 3.0, 2.4, "Rasa Server", "cổng 5005\nNLU: DIETClassifier\n(phân loại ý định)\nCore: RulePolicy, TEDPolicy\n(chọn hành động)", DARK)
box(ax, 10.6, 3.6, 2.5, 1.9, "Action Server", "cổng 5055\naction_rag_retrieval\n(Rasa SDK)", BLUE)
box(ax, 10.6, 0.3, 2.5, 1.7, "Câu trả lời mẫu", "utter_greet,\nutter_goodbye,\nutter_iamabot", GREEN)
box(ax, 13.8, 4.4, 2.1, 1.4, "ChromaDB", "413 đoạn\n(vector bge-m3)", ORANGE, fs=10)
box(ax, 13.8, 2.4, 2.1, 1.4, "Gemini API", "sinh câu trả lời\n(có model dự phòng)", ORANGE, fs=10)
arrow(ax, (2.4, 3.0), (3.2, 3.0), "hỏi")
arrow(ax, (5.6, 3.0), (6.4, 3.0), "REST", offset=(0, 0.2))
arrow(ax, (9.4, 3.7), (10.6, 4.4), "ask_admission_soict\nhoặc nlu_fallback", offset=(-0.3, 0.45))
arrow(ax, (9.4, 2.3), (10.6, 1.3), "greet / goodbye\n/ bot_challenge", offset=(-0.3, -0.4))
arrow(ax, (13.1, 5.0), (13.8, 5.1), "tìm", offset=(0, 0.2))
arrow(ax, (13.1, 4.0), (13.8, 3.2), "prompt", offset=(0.1, -0.25))
fig.savefig(OUT / "kien_truc_tong_the.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- 2. Luồng xử lý một câu hỏi (RAG) ----------------
fig, ax = canvas(13, 4.6)
steps = [
    ("1. Nhận câu hỏi", "tin nhắn mới nhất\n+ lịch sử 2 lượt", GREY),
    ("2. Viết lại câu hỏi", "Gemini đổi \"ngành này\"\nthành câu hỏi đầy đủ\n(query rewriting)", DARK),
    ("3. Embedding", "BAAI/bge-m3\nvector 1024 chiều\n(chuẩn hoá)", BLUE),
    ("4. Tìm kiếm", "ChromaDB\ncosine, top-10", ORANGE),
    ("5. Rerank", "Cross-encoder\nMiniLM-L-6, giữ top-5", BLUE),
    ("6. Sinh câu trả lời", "Gemini + prompt\nchỉ dùng ngữ cảnh,\nghi nguồn", RED),
]
w, gap = 1.85, 0.27
for i, (t, s, c) in enumerate(steps):
    x = 0.25 + i * (w + gap)
    box(ax, x, 1.3, w, 1.7, t, s, c, fs=10)
    if i:
        arrow(ax, (x - gap, 2.15), (x, 2.15))
ax.text(6.5, 0.55, "Không tìm thấy thông tin trong tài liệu  →  trả lời: \"Tôi không tìm thấy thông tin phù hợp...\""
        "   |   Gemini lỗi/quá tải  →  thử model dự phòng", ha="center", fontsize=9.5, color=GREY, style="italic")
fig.savefig(OUT / "luong_rag.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- 3. Pipeline nạp dữ liệu (offline) ----------------
fig, ax = canvas(13, 3.9)
steps = [
    ("Thu thập", "soict.hust.edu.vn\nts.hust.edu.vn\n(HTML, ảnh bảng)", GREY),
    ("Làm sạch", "bỏ menu, quảng cáo\ngiữ bảng, ghi nguồn\n→ Markdown", DARK),
    ("Tạo PDF", "24 file PDF\n(reportlab, font Arial)", RED),
    ("Đọc & cắt đoạn", "PyMuPDF, đọc bảng\ntheo hàng; đoạn\n800 ký tự, chồng 150", BLUE),
    ("Embedding", "bge-m3\n(normalize)", BLUE),
    ("Lưu ChromaDB", "collection\nsoict_admission\n413 đoạn", ORANGE),
]
for i, (t, s, c) in enumerate(steps):
    x = 0.25 + i * (w + gap)
    box(ax, x, 0.9, w, 2.0, t, s, c, fs=10)
    if i:
        arrow(ax, (x - gap, 1.9), (x, 1.9))
ax.text(6.5, 0.35, "scripts/ingest_documents.py — chạy 1 lần, chạy lại khi cập nhật tài liệu trong docs/admission/",
        ha="center", fontsize=9.5, color=GREY, style="italic")
fig.savefig(OUT / "pipeline_du_lieu.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- 4. Quy trình thực hiện dự án ----------------
fig, ax = canvas(13, 3.3)
steps = [("Khảo sát\n& thiết kế", GREY), ("Thu thập\ndữ liệu", DARK), ("Xây dựng\nNLU (Rasa)", RED),
         ("Xây dựng\nRAG", BLUE), ("Giao diện\nStreamlit", ORANGE), ("Kiểm thử\n& đánh giá", GREEN)]
for i, (t, c) in enumerate(steps):
    x = 0.25 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.9), w, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec=c))
    ax.text(x + w / 2, 1.65, t, ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.text(x + w / 2, 0.55, f"Bước {i + 1}", ha="center", fontsize=10, color=GREY)
    if i:
        arrow(ax, (x - gap, 1.65), (x, 1.65))
fig.savefig(OUT / "quy_trinh.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("ok")
