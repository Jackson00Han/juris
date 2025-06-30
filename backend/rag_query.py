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

# ─── CONFIG ────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME      = "all-MiniLM-L6-v2"
INDEX_PATH            = "laws_mini.faiss"
META_PATH             = "laws_mini_meta.json"
GEN_MODEL             = "google/flan-t5-large"            # updated model
CROSS_ENCODER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-12-v2"
INITIAL_TOP_K         = 20
RERANK_TOP_K          = 3     # reduced to 3 for brevity and context fit
CHUNK_SIZE            = 800

# Keywords for lightweight filtering
KEYWORD_MAPPING = {
    'refuse': 'afvise',
    'afvise': 'afvise',
}

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def load_index_and_meta(index_path: str, meta_path: str):
    idx = faiss.read_index(index_path)
    with open(meta_path, encoding="utf8") as f:
        meta = json.load(f)
    return idx, meta


def load_law_docs(json_dir: str) -> dict:
    docs = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith('.json') or fname == os.path.basename(META_PATH):
            continue
        data = json.load(open(os.path.join(json_dir, fname), encoding='utf8'))
        docs[data['id']] = data
    return docs


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

# Retrieve initial candidates via FAISS

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

# Lightweight keyword filtering

def keyword_filter(question: str, hits: list) -> list:
    query = question.lower()
    terms = [KEYWORD_MAPPING[k] for k in KEYWORD_MAPPING if k in query]
    if not terms:
        return hits
    filtered = [h for h in hits if any(term in load_text(h).lower() for term in terms)]
    return filtered if len(filtered) >= RERANK_TOP_K else hits

# Re-rank with cross-encoder

def rerank_chunks(question: str,
                  hits: list,
                  cross_encoder: CrossEncoder,
                  top_k: int):
    texts = [load_text(h) for h in hits]
    pairs = [[question, t] for t in texts]
    scores = cross_encoder.predict(pairs)
    scored = list(zip(hits, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [h for h, s in scored[:top_k]]

# Dynamic trimming to fit model max tokens

def trim_for_max_tokens(question: str, retrieved: list, tokenizer: AutoTokenizer, max_len: int) -> list:
    # measure prompt overhead tokens
    q_tokens = tokenizer.tokenize(question)
    overhead = len(tokenizer.tokenize("You are a Danish legal assistant.")) + 10
    # build list of (hit, token_length)
    hit_texts = [load_text(h) for h in retrieved]
    hit_tokens = [len(tokenizer.tokenize(text)) for text in hit_texts]
    selected = []
    total = len(q_tokens) + overhead
    for h, tkn_count in zip(retrieved, hit_tokens):
        if total + tkn_count <= max_len:
            selected.append(h)
            total += tkn_count
        else:
            break
    return selected

# Generate answer with dynamic context trimming

def generate_answer(question: str, retrieved: list, generator, tokenizer: AutoTokenizer) -> str:
    max_len = tokenizer.model_max_length
    trimmed = trim_for_max_tokens(question, retrieved, tokenizer, max_len)
    context = "\n\n".join(
        f"[Law {h['law_id']} {h['paragraph']} {h['section']}] {load_text(h)}"
        for h in trimmed
    )
    prompt = (
    "You are a Danish legal assistant. "
    "Based on the provided excerpts, answer in a complete sentence and cite the section. "
    "If unsure, say you don’t know.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {question}\n"
    "Answer:"
    )
    output = generator(prompt, max_length=512, do_sample=False)
    return output[0]['generated_text'].strip()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    global LAW_DOCS
    JSON_DIR = os.getenv('LAWS_JSON_DIR', 'laws_json')
    LAW_DOCS = load_law_docs(JSON_DIR)

    parser = argparse.ArgumentParser(description="RAG with dynamic trimming")
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("--meta",  default=META_PATH)
    parser.add_argument("--top-k", type=int, default=INITIAL_TOP_K)
    parser.add_argument("--rerank-k", type=int, default=RERANK_TOP_K)
    args = parser.parse_args()

    idx, meta = load_index_and_meta(args.index, args.meta)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    while True:
        try:
            q = input("Enter your legal question (or 'quit'): ")
        except EOFError:
            break
        if q.lower() in ("quit", "exit"):
            break

        hits = retrieve_chunks(q, idx, meta, embedder, args.top_k)
        hits = keyword_filter(q, hits)
        hits = rerank_chunks(q, hits, cross_encoder, args.rerank_k)

        print("Top hits:")
        for h in hits:
            print(f"- {h['law_id']} {h['paragraph']} (dist={h['distance']:.4f})")

        ans = generate_answer(q, hits, generator, tokenizer)
        print(f"Answer:\n{ans}\n")

if __name__ == "__main__":
    main()
