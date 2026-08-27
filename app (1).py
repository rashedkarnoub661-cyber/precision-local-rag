import streamlit as st
import os
import time
import torch
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="High Precision RAG", layout="wide")
st.title("🎯 AI Document Chatbot - Local GPU RAG")

# 1. تحميل النموذج محلياً على كارت الشاشة GPU لمنع مشاكل API نهائياً
@st.cache_resource
def load_local_llm():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return pipe

# تحميل النموذج عند بدء التطبيق مرة واحدة
llm_pipeline = load_local_llm()

# دالة مسح ملفات الـ PDF المؤقتة
def clean_workspace():
    for f in os.listdir():
        if f.endswith(".pdf"):
            try:
                os.remove(f)
            except:
                pass

# تهيئة مفتاح أداة الرفع في الـ session_state لضمان التصفير الكامل
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# زر إعادة التهيئة الشاملة وتصفير الواجهة والذاكرة
if st.button("♻️ Clear Memory & Start Fresh"):
    st.session_state.clear()
    clean_workspace()
    st.session_state.uploader_key = int(time.time())
    st.rerun()

# أداة رفع الملفات مربوطة بمفتاح ديناميكي لتفريغها فوراً عند الضغط على الزر
uploaded_file = st.file_uploader(
    "Upload your PDF",
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

        with st.spinner("Processing document in memory..."):
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # تقطيع نصوص دقيق (Precision Chunking)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,
                chunk_overlap=40
            )
            chunks = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # اسم مجموعة فريد لمنع تداخل الذاكرة تماماً بين الملفات
            unique_collection_name = f"col_{int(time.time())}"
            st.session_state.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=unique_collection_name
            )
            st.session_state.last_file = uploaded_file.name
            st.session_state.messages = []

        st.success("✅ File loaded cleanly into In-Memory ChromaDB!")

# عرض سجل المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# استقبال الأسئلة واسترجاع النصوص والتوليد
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.get("vectorstore"):
        # استرجاع دقيق باستخدام MMR
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10}
        )
        docs = retriever.invoke(prompt)
        context_text = "\n\n".join([d.page_content for d in docs])

        # تنسيق التعليمات للنموذج المحلي
        messages_format = [
            {
                "role": "system",
                "content": "You are a helpful and precise assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say 'I don't have enough information to answer this based on the document.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion:\n{prompt}"
            }
        ]

        with st.spinner("Generating answer with Local GPU Model..."):
            try:
                formatted_prompt = llm_pipeline.tokenizer.apply_chat_template(
                    messages_format, tokenize=False, add_generation_prompt=True
                )
                outputs = llm_pipeline(
                    formatted_prompt,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True
                )
                response = outputs[0]["generated_text"][len(formatted_prompt):].strip()
            except Exception as e:
                response = f"⚠️ Generation Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
            with st.expander("🔍 View Retrieved Context Chunks"):
                st.write(context_text)
    else:
        st.warning("Please upload a PDF file first!")
