
import streamlit as st
from utils import load_chunks, build_index, retrieve, generate_response

# 🛠 Page Setup
st.set_page_config(page_title="Loan Approval RAG Chatbot", layout="wide", page_icon="📊")

# 🌙 Dark Theme Styling
st.markdown("""
    <style>
        body {
            background-color: #1e1e2f;
            color: #f0f0f0;
        }
        .stApp {
            background-color: #1e1e2f;
        }
        h1 {
            text-align: center;
            color: #f8f8ff;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #cccccc;
            font-size: 18px;
            margin-top: 0px;
        }
        .stTextInput input {
            background-color: #2d2d44;
            color: #ffffff;
            border: 2px solid #4e8cff;
            border-radius: 10px;
            padding: 0.6rem;
            font-size: 18px;
        }
        .stButton>button {
            background-color: #4e8cff;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #3b6edb;
            transition: 0.3s;
        }
        .block-container {
            padding-top: 2rem;
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 3rem;
            font-size: 14px;
        }
        .context-box {
            background-color: #2a2a3d;
            color: #f0f0f0;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(255,255,255,0.1);
            font-family: monospace;
            font-size: 14px;
            white-space: pre-wrap;
        }
    </style>
""", unsafe_allow_html=True)

# 🔹 Header
st.markdown("<h1>📊 Loan Approval RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask questions from the dataset using retrieval-augmented generation</div>", unsafe_allow_html=True)

# 🧠 Load Embeddings
@st.cache_resource
def setup():
    chunks = load_chunks("data/Training Dataset.csv")
    index, all_chunks = build_index(chunks)
    return index, all_chunks

index, all_chunks = setup()

# 💬 Input Section (no random button)
st.markdown("### 💬 Ask your question:")

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

query = st.text_input(
    "",
    value=st.session_state.user_query,
    placeholder="e.g. What affects loan approval the most?",
    label_visibility="collapsed",
    key="user_query"
)

# 🤖 AI Response
if query:
    with st.spinner("🔍 Generating response..."):
        context = retrieve(query, index, all_chunks)
        answer = generate_response(context, query)

    st.markdown("### 💡 AI Answer")
    st.success(answer)

    with st.expander("📄 View Retrieved Context"):
        st.markdown(f"<div class='context-box'>{context}</div>", unsafe_allow_html=True)

# 🧾 Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit, FAISS & FLAN-T5</div>", unsafe_allow_html=True)
