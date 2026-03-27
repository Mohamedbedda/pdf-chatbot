import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker

from src.config import settings


_embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},  # Normalize for cosine similarity in FAISS
)


_reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)


def build_indexes(chunks: list[str]):
    """
    Build FAISS (vector) + BM25 (keyword) indexes from chunks.
    Called once at PDF upload — not per question.
    """
    embeddings = np.array(_embedder.embed_documents(chunks), dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    bm25 = BM25Okapi([c.split() for c in chunks])

    return index, bm25


def retrieve(question: str, chunks: list[str], index, bm25) -> list[dict] | None:
    """
    3-stage hybrid retrieval:
    1. FAISS : top TOP_K candidates by vector similarity
    2. BM25 : keyword boost (hybrid score = 0.7 * FAISS + 0.3 * BM25)
    3. Rerank : CrossEncoder scores top RERANK_TOP_K and thresholding with RERANK_THRESHOLD, then return FINAL_CONTEXT_K
    """

    q_emb = np.array([_embedder.embed_query(question)], dtype="float32")
    faiss_scores, indices = index.search(q_emb, settings.TOP_K)

    bm25_scores = bm25.get_scores(question.split())
    # Normalize BM25 scores to [0,1]
    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    bm25_range = bm25_max - bm25_min
    bm25_norm = (bm25_scores - bm25_min) / bm25_range if bm25_range > 0 else np.zeros_like(bm25_scores)

    candidates = []
    for score, idx in zip(faiss_scores[0], indices[0]):
        hybrid = 0.7 * float(score) + 0.3 * float(bm25_norm[idx])
        candidates.append({
            "chunk": chunks[idx],
            "similarity": hybrid,
        })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    top = candidates[:settings.RERANK_TOP_K]
    pairs = [(question, c["chunk"]) for c in top]

    rerank_scores = _reranker.compute_score(pairs, normalize=True)

    for c, s in zip(top, rerank_scores):
        c["rerank_score"] = float(s)

    top.sort(key=lambda x: x["rerank_score"], reverse=True)

    if top[0]["rerank_score"] < settings.RERANK_THRESHOLD:
        print(f"Rerank scores below threshold: {[c['rerank_score'] for c in top]}")
        return None 
    
    return top[:settings.FINAL_CONTEXT_K]


