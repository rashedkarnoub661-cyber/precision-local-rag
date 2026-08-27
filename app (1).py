import streamlit as st
import os
import time
from rag_engine import load_llm, build_vectorstore, generate_answer

st.set_page_config(page_title="High Precision RAG", layout="wide")
st.title("🎯 High-Precision Local RAG Engine")

# 1. تحميل النموذج محلياً على كارت الشاشة GPU
@st.cache_resource
def get_pipeline():
    return load_llm()

llm_pipeline = get_pipeline()

# دالة مسح ملفات الـ PDF المؤقتة
def clean_workspace():
    for f in os.listdir():
        if f.endswith(".pdf"):
            try:
                os.remove(f)
            except:
                pass

# تهيئة مفتاح أداة الرفع
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# زر إعادة التهيئة الشاملة وتصفير الذاكرة
if st.button("♻️ Clear Memory & Start Fresh"):
    st.session_state.clear()
    clean_workspace()
    st.session_state.uploader_key = int(time.time())
    st.rerun()

# أداة رفع الملفات بمفتاح ديناميكي
uploaded_file = st.file_uploader(
    "Upload your PDF Document", 
    type="pdf", 
    key=f"file_uploader_{st.session_state.uploader_key}"
)

# معالجة الملف المرفوع
if uploaded_file:
    if st.session_state.get("last_file") != uploaded_file.name:
        st.session_state.vectorstore = None
        clean_workspace()
        
        file_path = f"doc_{int(time.time())}.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing document & generating embeddings..."):
            st.session_state.vectorstore = build_vectorstore(file_path)
            st.session_state.last_file = uploaded_file.name
            st.session_state.messages = []
        
        st.success("✅ File loaded cleanly into ChromaDB Vectorstore!")

# عرض سجل المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# استقبال الأسئلة وتوليد الإجابات
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.get("vectorstore"):
        with st.spinner("Generating answer with Local GPU Model..."):
            try:
                response, context_text = generate_answer(
                    prompt, 
                    st.session_state.vectorstore, 
                    llm_pipeline,
                    st.session_state.messages
                )
            except Exception as e:
                response, context_text = f"⚠️ Error: {e}", ""

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
            if context_text:
                with st.expander("🔍 View Retrieved Context Chunks & Page Sources"):
                    st.write(context_text)
    else:
        st.warning("Please upload a PDF file first!")
