# Chatbot Tư vấn Tuyển sinh SoICT

Chatbot hỗ trợ thí sinh tra cứu thông tin tuyển sinh của Trường Công nghệ Thông tin và
Truyền thông (SoICT) - Đại học Bách khoa Hà Nội: ngành đào tạo, phương thức xét tuyển,
điểm chuẩn, học phí, học bổng...

## Kiến trúc

```
Người dùng ── Streamlit (8501) ── Rasa Server (5005) ── Action Server (5055)
                                    │  NLU: DIET                 │
                                    │  phân loại ý định          ├─ bge-m3: embedding câu hỏi
                                    │                            ├─ ChromaDB: tìm 10 đoạn gần nhất
                                    └─ greet/goodbye/bot →       ├─ Cross-encoder: rerank lấy 5 đoạn
                                       trả lời có sẵn            └─ Gemini: sinh câu trả lời + trích nguồn
```

- `ask_admission_soict` hoặc câu hỏi bot không chắc (`nlu_fallback`, độ tin cậy < 0.4) → `action_rag_retrieval`.
- Câu trả lời chỉ dựa trên tài liệu trong `docs/admission/`, có ghi nguồn (tên file, trang).

## 1) Yêu cầu

- **Python 3.10** (Rasa 3.6 không chạy trên Python 3.11+)
- API key Google Gemini (https://aistudio.google.com/apikey)

## 2) Cài đặt

Dùng **Python 3.10.11**. Các bản 3.10.12 trở lên đổi module `tarfile`, khiến Rasa 3.6 lỗi `WinError 123` khi mở model trên Windows.
Cách dễ nhất là dùng [uv](https://docs.astral.sh/uv/):

```powershell
uv python install 3.10.11
uv venv .venv --python 3.10.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv venv .venv-ui --python 3.10.11
uv pip install --python .venv-ui\Scripts\python.exe -r requirements-ui.txt
copy .env.example .env   # rồi điền GEMINI_API_KEY vào .env
```

Giao diện Streamlit dùng môi trường riêng (`.venv-ui`) để tránh xung đột thư viện với Rasa.

## 3) Nạp tài liệu tuyển sinh

Đặt tài liệu (`.pdf`, `.docx`, `.txt`, `.md`) vào `docs/admission/`, rồi chạy:

```powershell
python scripts/ingest_documents.py
```

PDF dạng ảnh scan cần OCR trước (hoặc chép nội dung sang `.txt`).

`docs/admission/` gồm 24 file PDF (86 trang, ~420 đoạn khi nạp) tổng hợp từ soict.hust.edu.vn và ts.hust.edu.vn
(truy cập 22/11/2026), mỗi file ghi URL nguồn ở đầu:

| Nhóm | File |
|---|---|
| Tổng hợp SoICT 2023-2026 | `soict_tong_hop_tuyen_sinh_2023_2026.pdf` (chỉ tiêu, điểm chuẩn 3 phương thức, học phí theo năm) |
| Thông tin tuyển sinh SoICT | `soict_thong_tin_tuyen_sinh_2024/2025/2026.pdf`, `soict_diem_chuan_2024_2026.pdf`, `soict_hoi_dap_tuyen_sinh.pdf`, `soict_gioi_thieu_kien_tao_tuong_lai.pdf` |
| Chương trình đào tạo | `chuong_trinh_IT1…`, `IT2`, `IT-E6`, `IT-E7`, `IT-E10`, `IT-E15`, `IT-EP`, `tai_nang_khoa_hoc_may_tinh` |
| ĐHBK Hà Nội | `hust_thong_tin_tuyen_sinh_2026.pdf`, `hust_de_an_tuyen_sinh_2024.pdf`, `hust_diem_chuan_2023…2026.pdf`, `hust_quy_doi_diem_chuan_2025.pdf`, `hust_hoc_phi_2026.pdf`, `hust_hoc_bong_ho_tro_tai_chinh.pdf` |

Bản markdown gốc của các PDF nằm trong `docs/tai_lieu_bo_sung/`. Script nạp đọc bảng trong PDF theo từng hàng
("Mã xét tuyển: IT1; 2025: 29.19; ...") để chatbot tra số liệu chính xác.

## 4) Train và chạy

```powershell
.\.venv\Scripts\python.exe -m rasa train --fixed-model-name soict-admission
```

Chạy cả 3 server (mỗi server một cửa sổ), rồi mở http://localhost:8501:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

Hoặc chạy tay, mỗi lệnh một terminal:

```powershell
.\.venv\Scripts\python.exe -m rasa run actions
.\.venv\Scripts\python.exe -m rasa run --enable-api --cors "*" --model models\soict-admission.tar.gz
.\.venv-ui\Scripts\python.exe -m streamlit run streamlit_app.py
```

Có thể chat trực tiếp trong terminal bằng `.\.venv\Scripts\python.exe -m rasa shell` thay cho 2 lệnh sau.

## 5) Kiểm thử

```powershell
rasa data validate        # kiểm tra domain / NLU / stories khớp nhau
rasa test nlu --cross-validation --folds 5   # đánh giá phân loại ý định -> results/
rasa test core --stories tests/test_stories.yml
```

## 6) Cấu trúc

- `data/`: dữ liệu NLU, stories, rules
- `domain.yml`: intents, câu trả lời, actions
- `config.yml`: pipeline NLU (tiếng Việt) và policies
- `actions/action_rag_retrieval.py`: custom action RAG
- `scripts/ingest_documents.py`: nạp tài liệu vào ChromaDB
- `docs/admission/`: tài liệu tuyển sinh nguồn
- `streamlit_app.py`: giao diện web
