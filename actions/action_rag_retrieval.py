import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Text

import chromadb
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Console Windows mặc định không phải UTF-8 -> tránh lỗi khi in tiếng Việt
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:  # Đọc biến môi trường từ file .env (nếu có)
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ==================== Configuration ====================
GEN_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# Thử lần lượt các model: model đầu quá tải (503) / hết hạn (404) / quá giờ thì chuyển model sau.
LLM_MODELS = [
    m.strip().replace("models/", "")
    for m in os.getenv(
        "GEMINI_MODEL", "gemini-flash-lite-latest,gemini-3.5-flash-lite,gemini-3.8-flash"
    ).split(",")
    if m.strip()
]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_TIMEOUT = 20  # giây cho mỗi model
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "soict_admission")
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

NOT_FOUND_ANSWER = (
    "Tôi không tìm thấy thông tin phù hợp trong tài liệu tuyển sinh. "
    "Bạn vui lòng liên hệ Ban tuyển sinh của Trường để được tư vấn chi tiết."
)
ERROR_ANSWER = "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn."

if not GEN_API_KEY:
    print("[WARNING] Chưa đặt GEMINI_API_KEY (xem README).")


def call_gemini(prompt: str) -> str:
    """Gọi Gemini qua REST, tự chuyển sang model dự phòng khi lỗi."""
    last_error = None
    for model in LLM_MODELS:
        try:
            r = requests.post(
                GEMINI_URL.format(model=model),
                headers={"x-goog-api-key": GEN_API_KEY},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2},
                },
                timeout=GEMINI_TIMEOUT,
            )
            if r.status_code != 200:
                last_error = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                print(f"[WARNING] Gemini {last_error}")
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            if text.strip():
                return text.strip()
            last_error = f"{model}: câu trả lời rỗng"
        except Exception as e:  # timeout, lỗi mạng, JSON lạ
            last_error = f"{model}: {e}"
            print(f"[WARNING] Gemini {last_error}")
    raise RuntimeError(f"Tất cả model Gemini đều lỗi. Lỗi cuối: {last_error}")

# Model được nạp một lần khi khởi động action server
print(f"[INFO] Đang nạp embedding model {EMBEDDING_MODEL} và reranker...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)
print(f"[INFO] ChromaDB: {CHROMA_DB_PATH} / collection '{CHROMA_COLLECTION_NAME}'")


def _load_collection() -> Optional[Any]:
    """Mở collection theo đường dẫn tuyệt đối để không phụ thuộc thư mục chạy."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        return client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"[WARNING] Không mở được collection '{CHROMA_COLLECTION_NAME}': {e}")
        return None


class ActionRAGRetrieval(Action):
    """Tìm tài liệu tuyển sinh trong ChromaDB và sinh câu trả lời bằng Gemini."""

    def name(self) -> Text:
        return "action_rag_retrieval"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_message = (tracker.latest_message.get("text") or "").strip()
        if not user_message:
            dispatcher.utter_message(text="Xin lỗi, tôi chưa hiểu câu hỏi của bạn.")
            return []

        history_list: List[str] = []
        user_messages_only: List[str] = []
        for event in tracker.events:
            text = event.get("text")
            if not text:
                continue
            if event.get("event") == "user":
                history_list.append(f"Người dùng: {text}")
                user_messages_only.append(text)
            elif event.get("event") == "bot":
                history_list.append(f"Bot: {text}")

        # 4 dòng gần nhất (bỏ câu hỏi hiện tại) ~ 2 lượt hỏi đáp
        chat_history_str = (
            "\n".join(history_list[-5:-1])
            if len(history_list) > 1
            else "Chưa có lịch sử trò chuyện."
        )

        # Ghép câu hỏi trước để truy vấn hiểu các câu hỏi nối tiếp ("ngành đó", "còn học phí?")
        search_query = user_message
        if len(user_messages_only) >= 2:
            search_query = f"{user_messages_only[-2]}. {user_message}"

        collection = _load_collection()
        if collection is None or collection.count() == 0:
            dispatcher.utter_message(
                text=(
                    "Xin lỗi, kho tài liệu tuyển sinh chưa sẵn sàng. "
                    "Hãy chạy `python scripts/ingest_documents.py` để nạp tài liệu."
                )
            )
            return []

        try:
            docs = self._search_chromadb(collection, search_query, top_k=10)
            docs = self._rerank_documents(search_query, docs, top_k=5)
            if not docs:
                dispatcher.utter_message(text=NOT_FOUND_ANSWER)
                return []
            context = self._build_context(docs)
            response = self._generate_answer(user_message, context, chat_history_str)
            dispatcher.utter_message(text=response)
        except Exception as e:
            print(f"[ERROR action_rag_retrieval]: {e}")
            dispatcher.utter_message(text=ERROR_ANSWER)

        return []

    def _search_chromadb(
        self, collection: Any, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        query_embedding = embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
        )

        docs = []
        if results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                meta = meta or {}
                docs.append(
                    {
                        "text": doc,
                        "source": meta.get("source"),
                        "page": meta.get("page"),
                    }
                )
        return docs

    def _rerank_documents(
        self, query: str, docs: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []
        scores = reranker.predict([[query, d["text"]] for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def _build_context(self, docs: List[Dict[str, Any]]) -> str:
        parts = []
        for i, d in enumerate(docs, start=1):
            src = d.get("source") or "tài liệu"
            if d.get("page"):
                src += f", trang {d['page']}"
            parts.append(f"[{i}] (Nguồn: {src})\n{d['text']}")
        return "\n\n".join(parts)

    def _generate_answer(self, query: str, context: str, chat_history: str) -> str:
        prompt = f"""Bạn là trợ lý tư vấn tuyển sinh của Trường Công nghệ Thông tin và Truyền thông (SoICT) - Đại học Bách khoa Hà Nội.
Nhiệm vụ: trả lời thí sinh và phụ huynh dựa trên tài liệu tuyển sinh được cung cấp.

YÊU CẦU:
- Chỉ dùng thông tin trong [Tài liệu]. KHÔNG tự bịa điểm chuẩn, học phí, chỉ tiêu, mã ngành hay mốc thời gian.
- Nếu tài liệu không có thông tin, trả lời đúng câu: "{NOT_FOUND_ANSWER}"
- Trả lời bằng tiếng Việt, thân thiện, ngắn gọn (3-6 câu); dùng gạch đầu dòng khi liệt kê ngành, phương thức, mốc thời gian.
- Nếu số liệu gắn với một năm cụ thể, nêu rõ năm đó.
- Cuối câu trả lời ghi nguồn ngắn gọn, ví dụ: "(Nguồn: <tên tài liệu>, trang X)".
- Dùng [Lịch sử trò chuyện] để hiểu các từ như "ngành đó", "còn học phí thì sao".

[Lịch sử trò chuyện]
{chat_history}

[Tài liệu]
{context}

Câu hỏi: {query}

Trả lời:
"""
        try:
            return call_gemini(prompt)
        except Exception as e:
            print(f"[ERROR] Gemini API call failed: {e}")
            return ERROR_ANSWER
