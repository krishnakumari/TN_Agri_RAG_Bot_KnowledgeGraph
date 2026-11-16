import base64
from pathlib import Path
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

# Fallback high-res paddy field / farming image URL if no local image is placed in the folder
DEFAULT_BG_URL = (
    "https://images.unsplash.com/photo-1500382017468-9049fed747ef"
    "?auto=format&fit=crop&w=1920&q=80"
)

st.set_page_config(
    page_title="TN Agriculture Schemes Assistant",
    page_icon="🌾",
    layout="wide",
)


def get_background_css(local_image_path: str = "farmer_bg.jpg") -> str:
    """Loads a local farmer image as Base64 if present; otherwise uses DEFAULT_BG_URL."""
    img_file = Path(local_image_path)
    if img_file.exists():
        encoded = base64.b64encode(img_file.read_bytes()).decode()
        ext = img_file.suffix.lstrip(".").lower() or "jpeg"
        bg_source = f"data:image/{ext};base64,{encoded}"
    else:
        bg_source = DEFAULT_BG_URL

    return f"""
    <style>
    /* Full-screen farmer background with dark readability overlay */
    [data-testid="stAppViewContainer"] {{
        background-image:
            linear-gradient(rgba(10, 28, 18, 0.72), rgba(10, 28, 18, 0.82)),
            url("{bg_source}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Transparent top header bar */
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}

    /* Frosted dark-green sidebar */
    [data-testid="stSidebar"] {{
        background: rgba(12, 34, 22, 0.88);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(134, 239, 172, 0.25);
    }}

    [data-testid="stSidebar"] * {{
        color: #f0fdf4 !important;
    }}

    /* Hero Banner Card */
    .hero-banner {{
        background: linear-gradient(135deg, rgba(20, 83, 45, 0.88), rgba(22, 101, 52, 0.78));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(134, 239, 172, 0.35);
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }}
    .hero-banner h1 {{
        color: #fef08a;
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
    }}
    .hero-banner p {{
        color: #dcfce7;
        margin: 0.4rem 0 0 0;
        font-size: 0.98rem;
    }}

    /* Glassmorphic Chat Message Cards */
    [data-testid="stChatMessage"] {{
        background: rgba(15, 38, 25, 0.84) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(134, 239, 172, 0.22);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        color: #f8fafc !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.28);
    }}

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {{
        color: #f1f5f9 !important;
    }}

    [data-testid="stChatMessage"] a {{
        color: #fde047 !important;
        font-weight: 600;
    }}

    /* Styled Sidebar Buttons */
    .stButton > button {{
        width: 100%;
        background: rgba(34, 197, 94, 0.22);
        color: #fef9c3 !important;
        border: 1px solid rgba(134, 239, 172, 0.4);
        border-radius: 10px;
        text-align: left;
        padding: 0.55rem 0.85rem;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        background: rgba(34, 197, 94, 0.42);
        border-color: #fde047;
    }}

    /* Bottom Chat Input Bar */
    [data-testid="stChatInput"] {{
        background: rgba(12, 34, 22, 0.92) !important;
        border: 1px solid rgba(134, 239, 172, 0.45) !important;
        border-radius: 14px !important;
    }}
    </style>
    """


st.markdown(get_background_css("farmer_bg.jpg"), unsafe_allow_html=True)

# Hero Header
st.markdown(
    """
    <div class="hero-banner">
        <h1>🌾 Tamil Nadu Agriculture & Farmers Welfare Assistant</h1>
        <p>Instant scheme eligibility, subsidies & district-level guidance • Powered by GraphRAG & NeMo Guardrails</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def send_query(user_question: str):
    """Sends a question to the FastAPI backend and appends both user & bot turns."""
    st.session_state.messages.append({"role": "user", "content": user_question})
    try:
        resp = requests.post(API_URL, json={"question": user_question}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "context": data.get("retrieval_context", []),
            "blocked": data.get("guardrail_blocked", False),
        })
    except requests.exceptions.ConnectionError:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Cannot reach the FastAPI server. Please ensure `uvicorn api:app --reload` is running on port 8000.",
            "context": [],
            "blocked": True,
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Error: {e}",
            "context": [],
            "blocked": True,
        })


# Sidebar with Quick-Click Sample Questions
with st.sidebar:
    st.markdown("### 🌱 Farmer Quick Queries")
    st.caption("Click any question below to test the bot:")

    sample_prompts = [
        "🌽 Maize farmer schemes in Coimbatore?",
        "🌾 Subsidy for certified seeds or gypsum?",
        "🏢 Which officer handles block-level applications?",
        "🏏 Who won yesterday's IPL cricket match?",
    ]

    for label in sample_prompts:
        clean_q = label.split(" ", 1)[1]
        if st.button(label, key=clean_q):
            st.session_state.pending_prompt = clean_q

    st.divider()
    st.markdown("### 🛡️ System Architecture")
    st.markdown(
        """
        - **Knowledge Graph:** `tn.gov.in` Agri Schemes
        - **Orchestration:** LangChain + OpenAI
        - **Guardrails:** NVIDIA NeMo Rails
        - **Backend:** FastAPI REST API
        """
    )

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "**Vanakkam! 🙏** I am your Tamil Nadu Agriculture Schemes Assistant.\n\n"
                "Ask me about crop subsidies, certified seeds, gypsum distribution, farmer training grants, "
                "or district-specific eligibility across Tamil Nadu."
            ),
            "context": [],
            "blocked": False,
        }
    ]

# Process Sidebar Quick-Click if triggered
if "pending_prompt" in st.session_state:
    queued = st.session_state.pop("pending_prompt")
    with st.spinner("🌾 Searching Tamil Nadu Agriculture Knowledge Graph..."):
        send_query(queued)
    st.rerun()

# Render Conversation History
for msg in st.session_state.messages:
    avatar = "🧑‍🌾" if msg["role"] == "user" else "🌾"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("blocked"):
            st.warning(msg["content"])
        else:
            st.markdown(msg["content"])

        if msg.get("context"):
            with st.expander(f"🔍 Inspect Knowledge Graph Evidence ({len(msg['context'])} blocks)"):
                for idx, block in enumerate(msg["context"], start=1):
                    st.code(block, language="text")

# Bottom Chat Input
if prompt := st.chat_input("Ask about schemes, seeds, subsidies, or your district..."):
    with st.spinner("🌾 Searching Tamil Nadu Agriculture Knowledge Graph..."):
        send_query(prompt)
    st.rerun()