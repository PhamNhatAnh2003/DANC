import base64
import uuid
from pathlib import Path

import requests
import streamlit as st

# ==================== Config ====================
RASA_WEBHOOK = "http://localhost:5005/webhooks/rest/webhook"
ASSETS = Path(__file__).resolve().parent / "assets"
HUST_RED = "#C8102E"

SUGGESTIONS = [
    "SoICT có những ngành nào?",
    "Điểm chuẩn IT1 qua các năm?",
    "Học phí ngành Khoa học dữ liệu và AI?",
    "Có những phương thức xét tuyển nào?",
    "Chỉ tiêu tuyển sinh năm 2026?",
    "IT-E6 học bằng tiếng gì?",
]

# ==================== Page Setup ====================
st.set_page_config(
    page_title="Tư vấn tuyển sinh SoICT - HUST",
    page_icon=str(ASSETS / "favicon_soict.png"),
    layout="centered",
)


@st.cache_data
def b64(name: str) -> str:
    return base64.b64encode((ASSETS / name).read_bytes()).decode()


# ==================== Theme (nền HUST + thẻ chat trắng) ====================
st.markdown(
    f"""
<style>
.stApp {{
    background:
        linear-gradient(rgba(150, 10, 25, .72), rgba(70, 0, 10, .85)),
        url("data:image/jpeg;base64,{b64('bg_hust.jpeg')}") center / cover fixed;
}}
/* Thanh trên cùng: nền đỏ mờ để nội dung cuộn lên bị che gọn */
[data-testid="stHeader"] {{
    background: rgba(110, 0, 14, .88);
    backdrop-filter: blur(6px);
}}
[data-testid="stHeader"] * {{ color: #fff !important; }}
/* Cột nội dung trắng ở giữa: vẽ bằng nền của vùng cuộn (cuộn theo nội dung, luôn dài hết trang),
   nên tin nhắn dài bao nhiêu cũng nằm trong nền trắng. --col = bề rộng cột (khớp .block-container). */
[data-testid="stAppScrollToBottomContainer"], [data-testid="stBottom"] > div {{
    --col: min(46rem, 100%);
    background: linear-gradient(to right,
        transparent calc(50% - var(--col) / 2), #fff calc(50% - var(--col) / 2),
        #fff calc(50% + var(--col) / 2), transparent calc(50% + var(--col) / 2));
}}
[data-testid="stAppScrollToBottomContainer"] {{ background-attachment: local; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 4.5rem; padding-bottom: 1rem; }}
.soict-banner {{ width: 100%; border-radius: 12px; display: block; }}
.soict-title {{ color: {HUST_RED}; font-weight: 800; font-size: 1.55rem; margin: .9rem 0 .1rem; }}
.soict-sub {{ color: #555; font-size: .95rem; margin-bottom: .4rem; }}
[data-testid="stChatMessage"] {{ border-radius: 14px; }}
[data-testid="stChatInput"] {{ border: 2px solid {HUST_RED}; border-radius: 14px; }}
section[data-testid="stSidebar"] {{ background: #fff; }}
section[data-testid="stSidebar"][aria-expanded="true"] {{ box-shadow: inset -4px 0 0 {HUST_RED}; }}
.stButton > button {{ border-color: {HUST_RED}; color: {HUST_RED}; border-radius: 20px; width: 100%; }}
.stButton > button:hover {{ background: {HUST_RED}; color: #fff; border-color: {HUST_RED}; }}
.contact {{ font-size: .85rem; color: #444; line-height: 1.6; }}
.contact b {{ color: {HUST_RED}; }}
</style>
""",
    unsafe_allow_html=True,
)

# ==================== Session State ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "pending" not in st.session_state:
    st.session_state.pending = None


# ==================== Helper ====================
def send_to_rasa(message: str) -> str:
    """Gửi tin nhắn tới Rasa REST API và trả về phản hồi."""
    try:
        payload = {"sender": st.session_state.session_id, "message": message}
        response = requests.post(RASA_WEBHOOK, json=payload, timeout=90)
        if response.status_code == 200:
            data = response.json()
            if data:
                return "\n\n".join(msg.get("text", "") for msg in data if msg.get("text"))
            return "_(Bot không có phản hồi)_"
        return f"❌ Lỗi kết nối Rasa (HTTP {response.status_code})"
    except requests.exceptions.ConnectionError:
        return "❌ Không kết nối được Rasa. Hãy chờ cửa sổ Rasa báo *Rasa server is up and running* rồi thử lại."
    except requests.exceptions.Timeout:
        return "⏱️ Hệ thống phản hồi quá lâu. Vui lòng thử lại."
    except Exception as e:
        return f"❌ Lỗi: {e}"


# ==================== Sidebar ====================
with st.sidebar:
    st.image(str(ASSETS / "logo_soict.png"), use_column_width=True)
    st.markdown("#### 💡 Câu hỏi gợi ý")
    for q in SUGGESTIONS:
        if st.button(q, key=f"sg_{q}"):
            st.session_state.pending = q
    st.divider()
    if st.button("🗑️ Cuộc trò chuyện mới", key="reset"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.markdown(
        """
<div class="contact">
<b>Trường Công nghệ Thông tin và Truyền thông</b><br>
Đại học Bách khoa Hà Nội<br>
🌐 <a href="https://soict.hust.edu.vn" target="_blank">soict.hust.edu.vn</a><br>
🎓 Tuyển sinh: <a href="https://ts.hust.edu.vn" target="_blank">ts.hust.edu.vn</a><br>
☎️ Ban Tuyển sinh: 024 3868 3408<br>
📍 Số 1 Đại Cồ Việt, Hai Bà Trưng, Hà Nội
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("Chatbot dùng Rasa + ChromaDB + Google Gemini. Thông tin chỉ mang tính tham khảo.")

# ==================== Header ====================
st.markdown(
    f'<img class="soict-banner" src="data:image/jpeg;base64,{b64("banner_soict_2026.jpeg")}" '
    'alt="SoICT - Bách khoa Open Day 2026">',
    unsafe_allow_html=True,
)
st.markdown('<div class="soict-title">Chatbot Tư vấn tuyển sinh SoICT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="soict-sub">Hỏi đáp về ngành học, phương thức xét tuyển, điểm chuẩn, chỉ tiêu, '
    "học phí của Trường CNTT&amp;TT - Đại học Bách khoa Hà Nội.</div>",
    unsafe_allow_html=True,
)

AVATARS = {"user": "🧑‍🎓", "assistant": str(ASSETS / "favicon_soict.png")}

if not st.session_state.messages:
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown(
            "Xin chào! 👋 Mình là trợ lý tư vấn tuyển sinh **SoICT - HUST**. "
            "Bạn có thể hỏi bất kỳ điều gì về tuyển sinh, hoặc chọn một câu hỏi gợi ý ở thanh bên trái."
        )

# Hiển thị lịch sử tin nhắn
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])

# Ô nhập liệu (hoặc câu hỏi gợi ý vừa bấm)
typed = st.chat_input("Hỏi về ngành học, xét tuyển, điểm chuẩn, học phí...")
user_input = typed or st.session_state.pending
st.session_state.pending = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Đang tra cứu tài liệu tuyển sinh..."):
            bot_reply = send_to_rasa(user_input)
        st.markdown(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
