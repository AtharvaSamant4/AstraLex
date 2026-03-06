<div align="center">

# ⚖️ AstraLex — AI-Powered Indian Law Assistant

**Deep-Research Agentic RAG Chatbot for Indian Law**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, full-stack AI chatbot that answers questions about Indian law using a multi-stage retrieval-augmented generation (RAG) pipeline with adaptive 3-tier routing, hybrid search, cross-encoder reranking, and real-time streaming responses.

[Features](#-features) · [Architecture](#-architecture) · [Tech Stack](#-tech-stack) · [Getting Started](#-getting-started) · [API Reference](#-api-reference) · [Screenshots](#-screenshots)

</div>

---

## ✨ Features

### 🧠 Intelligent RAG Pipeline
- **3-Tier Adaptive Routing** — Automatically classifies queries into **Fast** (1 LLM call), **Standard** (2 calls), or **Deep Research** (3+ calls) tiers based on complexity
- **Hybrid Retrieval** — Combines FAISS vector search (384-dim sentence embeddings) + BM25 lexical search with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision reranking of retrieved chunks
- **Deep Research Mode** — Multi-hop iterative retrieval with research planning, evidence graph construction, and query decomposition for complex legal questions
- **Query Rewriting** — LLM-powered query expansion for better retrieval coverage
- **Hallucination Verification** — Post-generation verification module that checks answer faithfulness against source chunks
- **Intent Classification** — Detects non-legal queries and responds appropriately

### 💬 ChatGPT-Style Conversational Interface
- **Persistent Chat Sessions** — Full conversation history stored in PostgreSQL
- **Conversation Memory** — Last N messages injected into the RAG pipeline for context-aware responses
- **Real-Time Streaming** — Server-Sent Events (SSE) for token-by-token response streaming
- **Document Upload** — Upload your own legal PDFs, DOCX, TXT, or JSON files for private RAG

### 🔐 Authentication & Security
- **JWT Authentication** — Secure signup/login with bcrypt password hashing
- **Per-User Data Isolation** — Each user's sessions, messages, and documents are isolated
- **Multi-API-Key Rotation** — 9 Gemini API keys with automatic rotation and exhaustion handling

### 📊 Analytics & Monitoring
- **Query Analytics** — Track most common queries, failure patterns, and retrieval performance
- **System Metrics** — Real-time latency monitoring, P95 tracking, tier distribution, user activity
- **Feedback System** — Thumbs-up/down on assistant responses

### 🎨 Modern Frontend
- **Marketing Website** — Home, How It Works, Datasets pages with animations
- **Chat Dashboard** — ChatGPT-style interface with sidebar, streaming messages, markdown rendering
- **Dark Theme** — Custom purple/indigo gradient design system
- **Responsive** — Mobile-friendly with collapsible sidebar

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Next.js Frontend                          │
│  Landing Pages │ Auth (Login/Signup) │ Chat Dashboard │ Upload   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼─────────────────────────────────────┐
│                      FastAPI Backend (REST)                       │
│  Auth │ Sessions │ Chat │ Streaming │ Documents │ Analytics      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    RAG Pipeline (Adaptive Routing)                │
│                                                                   │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │  FAST   │   │   STANDARD   │   │ DEEP RESEARCH│              │
│  │ 1 call  │   │   2 calls    │   │  3+ calls    │              │
│  │ Direct  │   │ Rewrite+Gen  │   │ Plan+Iterate │              │
│  └─────────┘   └──────────────┘   │ +Evidence    │              │
│                                    └──────────────┘              │
│                                                                   │
│  Hybrid Retrieval: FAISS (dense) + BM25 (sparse) → RRF Fusion   │
│  Cross-Encoder Reranking → Context Building → LLM Generation     │
│  Hallucination Verification → Source Attribution                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  FAISS Index │   │ Neon Postgres│   │ Google Gemini│
│ 161+ chunks  │   │  Sessions,   │   │  9 API keys  │
│ 384-dim vecs │   │  Messages,   │   │  Auto-rotate │
│ + BM25 index │   │  Documents   │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## 🛠 Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI + Uvicorn |
| LLM Provider | Google Gemini (9-key rotation) |
| Embeddings | `all-MiniLM-L6-v2` (384-dim, sentence-transformers) |
| Vector Store | FAISS (Flat L2 index) |
| Lexical Search | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Database | Neon PostgreSQL (serverless) |
| Auth | bcrypt + PyJWT |
| Streaming | Server-Sent Events (SSE) |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | Next.js 16 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS v4 |
| Animations | Framer Motion |
| Data Fetching | SWR |
| Markdown | react-markdown + remark-gfm |
| Icons | Lucide React |

### Legal Datasets (7 Acts, 161+ chunks)
- 🏛 **Constitution of India** — Fundamental rights, directive principles, amendments
- ⚖️ **Indian Penal Code (IPC)** — Criminal offences and penalties
- 📋 **Code of Criminal Procedure (CrPC)** — Criminal proceedings and procedures
- 💍 **Hindu Marriage Act** — Marriage, divorce, maintenance provisions
- 🤝 **Special Marriage Act** — Inter-faith and civil marriages
- 🚫 **Dowry Prohibition Act** — Anti-dowry laws and penalties
- 🛡 **Domestic Violence Act** — Protection against domestic violence

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL database (or [Neon](https://neon.tech) serverless)
- Google Gemini API key(s)

### 1. Clone the Repository

```bash
git clone https://github.com/AtharvaSamant4/AstraLex.git
cd AstraLex
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv chatbot
# Windows
chatbot\Scripts\Activate.ps1
# macOS/Linux
source chatbot/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API Keys (at least 1 required, up to 9 for rotation)
GEMINI_KEY_1=your_gemini_api_key_here
GEMINI_KEY_2=your_second_key_here
# ... up to GEMINI_KEY_9

# Neon PostgreSQL
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require

# JWT Secret
JWT_SECRET=your_jwt_secret_here
```

### 4. Build the Index

```bash
python build_index.py
```

This processes all 7 legal datasets, generates embeddings, and creates the FAISS + BM25 indices.

### 5. Start the Backend

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 6. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:3000` and connects to the backend at `http://localhost:8000`.

---

## 📡 API Reference

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/signup` | Create a new account |
| `POST` | `/auth/login` | Login and receive JWT |

### Chat Sessions
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/sessions` | Create a new chat session |
| `GET` | `/chat/sessions` | List all user sessions |
| `GET` | `/chat/sessions/{id}` | Get session with messages |
| `DELETE` | `/chat/sessions/{id}` | Delete a session |
| `PATCH` | `/chat/sessions/{id}` | Rename a session |
| `POST` | `/chat/sessions/{id}/message` | Send message (sync) |
| `POST` | `/chat/sessions/{id}/stream` | Send message (SSE stream) |
| `POST` | `/chat/sessions/{id}/feedback` | Submit feedback |

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload PDF/DOCX/TXT/JSON |
| `GET` | `/documents` | List user's documents |
| `DELETE` | `/documents/{id}` | Delete a document |

### Analytics & Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/metrics` | System monitoring metrics |
| `GET` | `/analytics/top-queries` | Most common queries |
| `GET` | `/analytics/failure-queries` | Failed/low-quality queries |
| `GET` | `/analytics/retrieval-performance` | Retrieval stats |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Index statistics |

---

## 📁 Project Structure

```
AstraLex/
├── api/
│   └── server.py              # FastAPI REST API (v5.0.0)
├── rag/
│   ├── pipeline.py            # Main RAG pipeline with 3-tier routing
│   ├── model_manager.py       # 9-key Gemini rotation manager
│   ├── retriever.py           # FAISS vector retrieval
│   ├── bm25_search.py         # BM25 lexical retrieval
│   ├── reranker.py            # Cross-encoder reranking
│   ├── embedder.py            # Sentence-transformer embeddings
│   ├── chunker.py             # Document chunking
│   ├── context_builder.py     # RRF fusion + context assembly
│   ├── query_rewriter.py      # LLM query rewriting
│   ├── research_planner.py    # Deep research planning
│   ├── retrieval_loop.py      # Multi-hop iterative retrieval
│   ├── evidence_graph.py      # Evidence graph construction
│   ├── intent_classifier.py   # Legal/non-legal intent detection
│   ├── verifier.py            # Hallucination verification
│   ├── prompt_template.py     # System prompts for all tiers
│   ├── vectordb.py            # FAISS index management
│   └── loader.py              # JSON dataset loader
├── auth/                       # JWT + bcrypt authentication
├── database/                   # PostgreSQL schema & queries
├── services/                   # Chat service layer
├── documents/                  # Document upload & processing
├── analytics/                  # Query analytics module
├── metrics/                    # System monitoring metrics
├── data/                       # 7 Indian law JSON datasets
├── index/                      # Pre-built FAISS + BM25 indices
├── frontend/
│   └── src/
│       ├── app/               # Next.js App Router pages
│       │   ├── page.tsx       # Home (landing)
│       │   ├── how-it-works/  # Pipeline explanation
│       │   ├── datasets/      # Dataset showcase
│       │   ├── login/         # Login page
│       │   ├── signup/        # Signup page
│       │   └── chat/          # Chat dashboard (protected)
│       ├── components/
│       │   ├── landing/       # Navbar, Hero, Features, etc.
│       │   └── chat/          # Sidebar, ChatWindow, Message, etc.
│       ├── hooks/             # useChat, useStreaming, useDocuments
│       ├── context/           # AuthContext provider
│       ├── lib/               # API client, utilities
│       └── types/             # TypeScript interfaces
├── build_index.py              # Index builder script
├── cli_chat.py                 # CLI chat interface
├── eval_suite.py               # Evaluation suite (246 questions)
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables (not committed)
```

---

## 📊 Evaluation

The system has been evaluated on a suite of **246 legal questions** across all 7 acts:

| Metric | Score |
|--------|-------|
| **Overall Quality** | **8.8 / 10** |
| Retrieval Success Rate | 95%+ |
| Source Attribution | Cited in every response |
| Tier Distribution | ~40% Fast, ~35% Standard, ~25% Deep |

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

## ⚠️ Disclaimer

AstraLex is an AI research tool. It is **not a substitute for professional legal advice**. Always consult a qualified legal professional for actual legal matters. The information provided may be incomplete or inaccurate.

---

<div align="center">

Built with ❤️ by [Atharva Samant](https://github.com/AtharvaSamant4)

</div>
