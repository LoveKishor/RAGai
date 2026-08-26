# 🧠 EV Assistant

A Retrieval-Augmented Generation (RAG) chatbot for EV (electric vehicle) diagnostics, built with **Streamlit**, **Pinecone**, **LangChain**, and **Groq**. Upload technical PDFs, ask diagnostic questions, and get grounded answers backed only by your knowledge base — with a self-improving memory that learns from corrections over time.

Live demo: [evdiagnosis.streamlit.app](https://evdiagnosis.streamlit.app/)

---

## ✨ Features

- **Document-grounded Q&A** — answers are generated only from uploaded PDF content and past corrections, with an explicit "insufficient information" fallback to avoid hallucination.
- **HyDE retrieval (optional)** — generates a hypothetical answer passage to improve semantic search recall before hitting the vector store.
- **Cross-encoder reranking** — retrieved chunks are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` for higher-precision context selection.
- **Self-learning memory** — 👍/👎 feedback on answers lets admins save corrections into a dedicated Pinecone namespace, which is retrieved alongside document context on future queries.
- **Password-protected admin controls** — separate gates for PDF uploads and feedback/correction management, backed by an admin Groq key.
- **Bring-your-own-key mode** — non-admin users can supply their own Groq API key to chat without needing upload/feedback access.
- **Dynamic model selection** — fetches and lists all Groq models available to the active API key.
- **Automatic Pinecone index bootstrap** — creates the serverless index on first run if it doesn't already exist.

---

## 🏗️ How It Works

1. **Ingestion** — Admins upload PDFs via the sidebar. Each document is split into chunks (`RecursiveCharacterTextSplitter`), embedded with a HuggingFace sentence-transformer model, and upserted into a Pinecone `pdfs` namespace with deterministic, hash-based IDs (so duplicate uploads don't create duplicate vectors).
2. **Retrieval** — On each user question, relevant chunks are retrieved from Pinecone (optionally via a HyDE-generated hypothetical passage), then reranked with a cross-encoder to surface the most relevant results.
3. **Memory recall** — Past user-confirmed answers and corrections are retrieved from a separate `memory` namespace and passed to the LLM as additional context.
4. **Generation** — A Groq-hosted LLM (default: `llama3-70b-8192`) generates an answer strictly from the retrieved context and memory, following a diagnostic-style prompt (diagnosis → evidence → next checks → facts vs. recommendations).
5. **Feedback loop** — When feedback mode is enabled, 👍 saves the Q&A pair to memory as-is; 👎 prompts the admin for the correct answer, which is then saved to memory for future retrieval.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Orchestration | [LangChain](https://www.langchain.com/) |
| Vector database | [Pinecone](https://www.pinecone.io/) (serverless) |
| Embeddings | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2` by default) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM inference | [Groq](https://groq.com/) |
| PDF parsing | `pypdf` via `PyPDFLoader` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- A [Pinecone](https://www.pinecone.io/) account and API key
- A [Groq](https://console.groq.com/) account and API key

### Installation

```bash
git clone https://github.com/<your-username>/ragai.git
cd ragai
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root (or configure Streamlit Secrets if deploying to Streamlit Cloud):

```env
PINECONE_API_KEY=your-pinecone-api-key
GROQ_API_KEY=your-groq-api-key
UPLOAD_PASSWORD=choose-an-admin-password

# Optional overrides
PINECONE_INDEX_NAME=self-learning-rag
GROQ_MODEL=llama3-70b-8192
EMBEDDING_MODEL=all-MiniLM-L6-v2
RETRIEVE_K=8
RERANK_K=4
MEMORY_K=2
USE_HYDE=false
```

### Run locally

```bash
streamlit run appv4.py
```

The app will automatically create the Pinecone index (384-dimensional, cosine similarity, AWS `us-east-1` serverless) on first launch if it doesn't already exist.

---

## 🔐 Access Modes

| Mode | Requirement | Capabilities |
|---|---|---|
| **Guest chat** | Own Groq API key | Ask questions using the existing knowledge base |
| **Upload admin** | Upload password | Upload PDFs to expand the knowledge base, plus chat with the admin Groq key |
| **Feedback admin** | Admin password | Enable/disable 👍/👎 feedback and save corrections to memory |

---

## 📁 Project Structure

```
ragai/
├── appv4.py           # Main Streamlit application
├── requirements.txt    # Python dependencies
└── .env                # Local secrets (not committed)
```

---
## 📈 Potential Improvements (for future scope)

- **Multi-user memory namespaces** – each user could have their own memory.
- **Conversation history** – use past messages as context.
- **Support for more file types** – Word, Excel, images (via OCR).
- **HyDE fine-tuning** – adjust the hypothetical prompt for better results.
- **Add logging and monitoring** – track usage, errors.
- **User authentication** – using Streamlit's built-in `st.login` (when stable).

