#!/usr/bin/env python3
import os
# Limit threading to avoid segfaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# ─── CONFIG (IMPROVED) ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME      = "all-MiniLM-L6-v2"                # unchanged embedding model
INDEX_PATH            = "laws_mini.faiss"
META_PATH             = "laws_mini_meta.json"
GEN_MODEL             = "google/flan-t5-base"            # local generation model
CROSS_ENCODER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # added cross-encoder for reranking
INITIAL_TOP_K         = 10    # Improved: increased initial retrieval from 5 to 10
RERANK_TOP_K          = 5     # Improved: select top 5 after cross-encoder rerank
CHUNK_SIZE            = 800   # must match ingest chunk size

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def load_index_and_meta(index_path: str, meta_path: str):
    idx = faiss.read_index(index_path)
    with open(meta_path, encoding="utf8") as f:
        meta = json.load(f)
    return idx, meta

# Load full JSON laws into memory for retrieving chunk text
# This supports better prompting by including actual legislative language

def load_law_docs(json_dir: str) -> dict:
    docs = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith('.json') or fname == os.path.basename(META_PATH):
            continue
        path = os.path.join(json_dir, fname)
        data = json.load(open(path, encoding='utf8'))
        docs[data['id']] = data
    return docs

# Extract actual text chunk from law JSON based on metadata
# Used to build enriched context for generation

def load_text(hit: dict) -> str:
    law = LAW_DOCS.get(hit['law_id'], {})
    for chap in law.get('structured_text', []):
        for para in chap.get('paragraphs', []):
            if para.get('paragraph') == hit['paragraph']:
                for sec in para.get('sections', []):
                    if sec.get('section') == hit['section']:
                        full = sec.get('text', '')
                        start = hit.get('char_offset', 0)
                        return full[start:start + CHUNK_SIZE]
    return ''

# Retrieve initial candidates via FAISS (IMPROVED TOP_K)

def retrieve_chunks(question: str,
                    idx: faiss.IndexFlatL2,
                    meta: list,
                    embedder: SentenceTransformer,
                    top_k: int):
    q_emb = embedder.encode([question], show_progress_bar=False)
    q_vec = np.array(q_emb, dtype="float32")
    distances, indices = idx.search(q_vec, top_k)
    hits = []
    for dist, i in zip(distances[0], indices[0]):
        info = meta[i].copy()
        info["distance"] = float(dist)
        hits.append(info)
    return hits

# Re-rank candidates with cross-encoder (NEW)

def rerank_chunks(question: str,
                  hits: list,
                  cross_encoder: CrossEncoder,
                  top_k: int):
    texts = [load_text(h) for h in hits]
    pairs = [[question, t] for t in texts]
    scores = cross_encoder.predict(pairs)
    # combine and sort by score descending
    scored = list(zip(hits, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    reranked = [h for h, s in scored[:top_k]]
    return reranked

# Generate answer using local model with better prompting (includes chapter)

def generate_answer(question: str, retrieved: list, generator) -> str:
    # Improved: include chapter, paragraph and actual text in context
    context = "\n\n".join(
        f"[Law {h['law_id']} Chapter {h.get('chapter','')} {h['paragraph']} {h['section']}] {load_text(h)}"
        for h in retrieved
    )
    prompt = (
        "You are a Danish legal assistant. "
        "Answer the user question based solely on the following legislative excerpts. "
        "If the answer is not contained within them, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    output = generator(prompt, max_length=512, do_sample=False)
    return output[0]['generated_text'].strip()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    global LAW_DOCS
    # load laws JSON for chunk text
    JSON_DIR = os.getenv('LAWS_JSON_DIR', 'laws_json')
    print(f"Loading law JSON files from {JSON_DIR}...")
    LAW_DOCS = load_law_docs(JSON_DIR)

    parser = argparse.ArgumentParser(description="RAG: retrieve & answer (with rerank)")
    parser.add_argument("--index", default=INDEX_PATH, help="FAISS index path")
    parser.add_argument("--meta",  default=META_PATH,  help="metadata JSON path")
    # Flags for improved retrieval configuration
    parser.add_argument("--top-k", type=int, default=INITIAL_TOP_K,
                        help="Number of chunks to retrieve initially (IMPROVED)")
    parser.add_argument("--rerank-k", type=int, default=RERANK_TOP_K,
                        help="Number of chunks after re-ranking (NEW)")
    args = parser.parse_args()

    print("Loading index and metadata...")
    idx, meta = load_index_and_meta(args.index, args.meta)
    print(f"Index has {idx.ntotal} vectors, dimension {idx.d}")

    # Initialize models
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"Loading generator model: {GEN_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    while True:
        try:
            q = input("\nEnter your legal question (or 'quit'): ")
        except EOFError:
            break
        if q.lower() in ("quit", "exit"):
            break

        # retrieve and rerank pipeline (IMPROVED)
        initial_hits = retrieve_chunks(q, idx, meta, embedder, args.top_k)
        hits = rerank_chunks(q, initial_hits, cross_encoder, args.rerank_k)

        print("\nTop retrieved chunks after rerank:")
        for h in hits:
            print(f"- Law {h['law_id']} {h['paragraph']} (offset {h['char_offset']}): dist={h['distance']:.4f}")

        ans = generate_answer(q, hits, generator)
        print(f"\nAnswer:\n{ans}\n")

if __name__ == "__main__":
    main()
