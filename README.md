# 🎯 Precision Local RAG Engine with Memory & GPU Acceleration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/rashedkarnoub661-cyber/precision-local-ragblob/main/RAG_Notebook.ipynb)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Community-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)

A high-precision Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangChain**, **ChromaDB**, and a local **Qwen2.5-1.5B-Instruct** LLM running natively on GPU.

Designed to prevent hallucinations, avoid API limits, and maintain context over multi-turn conversations.

---

## 🌟 Key Features

- **100% Local Inference:** Runs LLM (`Qwen2.5-1.5B`) directly on local GPU (CUDA) with zero external API dependencies.
- **Precision Chunking:** Uses `RecursiveCharacterTextSplitter` with small chunk size (250 chars / 40 overlap) optimized for dense structured PDFs (CVs, Anabin degree evaluations, technical papers).
- **MMR Retrieval:** Max Marginal Relevance search via ChromaDB ensures context diversity and reduces redundant chunks.
- **Multi-Turn Chat History:** Integrates contextual memory across follow-up queries without payload bloat.
- **Clean Workspace Manager:** Instant memory reset and file-uploader dynamic key reset mechanism.

---

## 🏗️ System Architecture

```text
[ PDF Upload ]
      │
      ▼
[ PyPDFLoader ] ──► [ Precision Chunking (250/40) ]
                              │
                              ▼
                 [ MiniLM-L6-v2 Embeddings ]
                              │
                              ▼
                   [ ChromaDB Vector Store ]
                              │
            ┌─────────────────┴─────────────────┐
            │ MMR Retrieval (k=4, fetch_k=10)   │
            └─────────────────┬─────────────────┘
                              ▼
            [ Local Qwen2.5-1.5B Model on GPU ] ──► [ Precise Response ]
