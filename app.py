# app.py
import os, json
import streamlit as st
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from duckduckgo_search import DDGS

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="🧠 AI Assistant", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 Modern AI Chatbot Assistant</h1>", unsafe_allow_html=True)

# ================== SESSION STATE ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "voice_trigger" not in st.session_state:
    st.session_state.voice_trigger = False

# ================== SAVE & LOAD CHAT ==================
def save_chat_history():
    try:
        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving chat: {e}")

def load_chat_history():
    try:
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r", encoding="utf-8") as f:
                st.session_state.chat_history = json.load(f)
    except Exception as e:
        st.error(f"Error loading chat: {e}")

load_chat_history()

# ================== VOICE INPUT ==================
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        st.success(f"👉 You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("❌ Could not understand speech")
        return ""
    except sr.RequestError:
        st.error("⚠️ Speech API unavailable")
        return ""

# ================== IMAGE SEARCH ==================
def search_image(query: str, num_results: int = 1):
    try:
        with DDGS() as ddgs:
            results = ddgs.images(query, max_results=num_results)
            return [r["image"] for r in results] if results else []
    except Exception as e:
        st.error(f"⚠️ Image search failed: {e}")
        return []

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("🔧 Settings")
    api_key = st.text_input("🔑 Enter Groq API Key (starts with gsk_)", type="password")
    if api_key:
        st.session_state.api_key = api_key.strip()

    model_choice = st.selectbox(
        "🤖 Model Selection",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )
    temperature = st.slider("🌡️ Creativity", 0.0, 2.0, 0.7, 0.1)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            save_chat_history()
            st.rerun()
    with c2:
        if st.button("💾 Save Chat", use_container_width=True):
            save_chat_history()
            st.success("Chat saved!")

    if st.button("🗑️ Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared. Please rerun app.")

# ================== LLM CLIENT ==================
@st.cache_resource(show_spinner=False)
def get_llm_client(api_key: str, model: str, temp: float):
    if not api_key:
        return None
    try:
        return ChatGroq(groq_api_key=api_key, model_name=model, temperature=temp)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

# ================== DISPLAY CHAT HISTORY ==================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])

# ================== USER INPUT ==================
col1, col2 = st.columns([3, 1])
user_prompt = None
with col1:
    user_prompt = st.chat_input("💬 Type your message here...")
with col2:
    if st.button("🎤 Speak"):
        voice_input = listen()
        if voice_input:
            user_prompt = voice_input
            st.session_state.voice_trigger = True

# ================== BOT RESPONSE ==================
if user_prompt or st.session_state.voice_trigger:
    st.session_state.voice_trigger = False
    if not st.session_state.api_key:
        st.error("🔑 Please enter your Groq API key in sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_prompt)

    llm = get_llm_client(st.session_state.api_key, model_choice, temperature)
    if not llm:
        st.error("❌ Failed to initialize AI model. Check API key.")
    else:
        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            try:
                with st.spinner("🤔 Thinking..."):
                    resp = llm.invoke([HumanMessage(content=user_prompt)])
                    full = resp.content
                msg_placeholder.markdown(full)
                st.session_state.chat_history.append({"user": user_prompt, "bot": full})
                st.session_state.last_response = full
                save_chat_history()
            except Exception as e:
                err = f"❌ Error generating response: {e}"
                msg_placeholder.error(err)
                st.session_state.chat_history.append({"user": user_prompt, "bot": err})

# ================== IMAGE SEARCH ==================
st.subheader("🔍 Image Search")
search_query = st.text_input("Enter something to search for images")
if st.button("Search Image"):
    if search_query:
        with st.spinner("🔎 Searching images..."):
            image_urls = search_image(search_query, num_results=3)
            if image_urls:
                st.success("✅ Found images:")
                for url in image_urls:
                    st.image(url, caption=search_query, use_column_width=True)
            else:
                st.warning("❌ No images found.")

# ================== FOOTER ==================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "💡 Tip: Enter API Key, choose a model, try voice input!"
    "</div>",
    unsafe_allow_html=True
)
