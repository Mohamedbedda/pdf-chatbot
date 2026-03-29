---
title: PDF Chatbot
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.10.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 📄 PDF Chatbot — RAG Pipeline

A production-grade **Retrieval-Augmented Generation (RAG) chatbot** that lets you
upload any PDF and ask natural language questions about its content.
Every answer is strictly grounded in the document — no hallucinations.

Built with **Python · Gradio · Groq · FAISS · BM25 · BGE Reranker**

---

## ✨ Features

- 📤 Upload any PDF and chat with it instantly
- 🔍 Hybrid retrieval: semantic (FAISS) + keyword (BM25 Okapi)
- 🎯 Cross-encoder reranking for precision (BGE FlagReranker)
- 🤖 4 selectable Groq LLM models
- 💬 Persistent chat history per session
- 🔒 Hallucination safe: returns no answer when context is insufficient

---

## 🔄 Retrieval Pipeline

```
                User Query
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  STAGE 1 — FAISS                            │
│  Embed query → cosine similarity search     │
│  Output: TOP_K = 10 similar chunks          │
└──────────────────────────────────────────── ┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  STAGE 2 — HYBRID BLEND                     │
│  hybrid = 0.7 × FAISS + 0.3 × BM25_norm     │
│  Output: top 7 chunks                       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STAGE 3 — FLAGRERANKER                     │
│  BGE cross-encoder scores (query, chunk)    │
│  If best score < threshold → return None    │
│  Output: best 3 chunks                      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
            Groq LLM → Answer
```

---

## 🔍 How Retrieval Works

### Stage 1 — FAISS (Semantic Search)
Chunks are embedded with `BAAI/bge-large-en-v1.5` into 1024-dimensional vectors
at PDF upload time. At query time, the question is embedded the same way and
FAISS performs a cosine similarity search to retrieve the 10 most semantically
relevant chunks.

### Stage 2 — Hybrid Blend (FAISS + BM25)
FAISS captures *meaning* but can miss exact keyword matches (names, article
numbers, acronyms). BM25 Okapi fills that gap with term-frequency keyword
scoring. Both scores are normalized to `[0, 1]` before blending:

```
hybrid_score = 0.7 × FAISS_score + 0.3 × BM25_norm
```

Because the weights sum to 1.0, the hybrid score is always in `[0, 1]`.

### Stage 3 — FlagReranker (Precision Filter)
FAISS and BM25 encode the query and chunks *independently*. The BGE cross-encoder
takes each **(query, chunk) pair together**, letting every query token attend to
every chunk token — producing a much deeper relevance score.

`compute_score(pairs, normalize=True)` applies a sigmoid to map scores to `[0, 1]`.
If even the best chunk scores below `RERANK_THRESHOLD (0.5)`, the system
returns `None` and the LLM will not generate a hallucinated answer.

---

## 📁 Project Structure

```
pdf-chatbot/
├── app.py                  ← Gradio UI entry point
├── src/
│   ├── config.py           ← All constants + Groq model configs
│   ├── state.py            ← Per-session gr.State()
│   └── rag/
│       ├── __init__.py
│       ├── pdf_loader.py   ← PyMuPDF extraction + 2-column handling
│       ├── retriever.py    ← FAISS + BM25 + BGE FlagReranker
│       └── generator.py    ← Groq LLM generation
```

---

## ⚙️ Key Configuration

```python
TOP_K: int = 10                  # FAISS retrieves 10 candidates
RERANK_TOP_K: int = 5            # FlagReranker re-scores top 5
FINAL_CONTEXT_K: int = 3         # Best 3 chunks passed to LLM
MAX_TOKENS: int = 500            # LLM max output length
RERANK_THRESHOLD: float = 0.7   # Minimum rerank score to return an answer
```

---

## 🤖 Supported LLM Models (via Groq)

| Model | ID | Characteristic |
|---|---|---|
| `llama-3.1-8b` | `llama-3.1-8b-instant` | 🚀 Fastest · low latency · high throughput |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | 🧠 Reasoning · MoE architecture · agentic tasks |
| `kimi-k2` | `moonshotai/kimi-k2-instruct` | 🔀 High throughput · long context |
| `qwen3-32b` | `qwen/qwen3-32b` | ⚡ Strong reasoning · fast inference |

All models share the same retrieval layer — only generation varies.


---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Gradio |
| PDF Parsing | PyMuPDF (`fitz`) |
| Chunking | LangChain `TextSplitter` |
| Embeddings | `BAAI/bge-large-en-v1.5` |
| Vector Index | FAISS `IndexFlatIP` |
| Keyword Search | BM25 Okapi (`rank_bm25`) |
| Reranker | `BAAI/bge-reranker-v2-m3` (`FlagEmbedding`) |
| LLM Inference | Groq API |

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Mohamedbedda/pdf-chatbot.git
cd pdf-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
create a .env file with GROQ_API_KEY=your_api_key_here

# 4. Run the app
python app.py
```