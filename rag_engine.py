import os
import time
import torch
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_llm():
    """تحميل نموذج Qwen2.5 محلياً على GPU"""
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return pipe

def build_vectorstore(file_path):
    """تحميل المستند وتقطيعه وإنشاء قاعدة بيانات ChromaDB"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    unique_collection_name = f"col_{int(time.time())}"
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=unique_collection_name
    )
    return vectorstore

def generate_answer(prompt, vectorstore, llm_pipeline, chat_history=None):
    """استرجاع النصوص المترابطة وتوليد الإجابة مع دعم الذاكرة التفاعلية وأرقام الصفحات"""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )
    docs = retriever.invoke(prompt)
    
    # استخراج النصوص مع أرقام الصفحات المصدرية
    context_text = "\n\n".join([f"[Page {d.metadata.get('page', 0) + 1}]: {d.page_content}" for d in docs])
    
    # صياغة الذاكرة التفاعلية (آخر جولتين من المحادثة)
    history_context = ""
    if chat_history and len(chat_history) > 1:
        recent_history = chat_history[-5:-1]
        history_context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_history])
    
    system_instruction = (
        "You are a helpful and precise assistant. Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't have enough information to answer this based on the document.'"
    )
    
    user_content = f"Context:\n{context_text}\n\n"
    if history_context:
        user_content += f"Previous Conversation History:\n{history_context}\n\n"
    user_content += f"Current Question:\n{prompt}"

    messages_format = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]

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
    return response, context_text