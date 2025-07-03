import os
# Limit threading to avoid segmentation faults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
from typing import Optional
import yaml, os, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from logger_setup import logger

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Initialize FastAPI
app = FastAPI(
    title="Danish Law RAG API",
    description="Retrieve Danish legal answers with citations",
)

# Load index & metadata once at startup
INDEX = faiss.read_index(cfg['index_path'])
with open(cfg['meta_path'], encoding='utf8') as f:
    META = json.load(f)

# Load sentence embedder & cross-encoder & generator
EMBEDDER = SentenceTransformer(cfg['embed_model'])
CROSS_ENCODER = CrossEncoder(cfg['cross_encoder_model'])
TOKENIZER = AutoTokenizer.from_pretrained(cfg['gen_model'])
GEN_MODEL = AutoModelForSeq2SeqLM.from_pretrained(cfg['gen_model'])
GEN_PIPE = pipeline("text2text-generation", model=GEN_MODEL, tokenizer=TOKENIZER)

# Load full law documents into a dict for text retrieval
import glob
LAW_DOCS = {}
for path in glob.glob(os.path.join(cfg['out_dir'], '*.json')):
    doc = json.load(open(path, encoding='utf8'))
    LAW_DOCS[doc['id']] = doc

# Helper: extract the chunk text given metadata
def load_text(hit: dict) -> str:
    law = LAW_DOCS.get(hit['law_id'], {})
    for chap in law.get('structured_text', []):
        for para in chap.get('paragraphs', []):
            if para.get('paragraph') == hit['paragraph']:
                for sec in para.get('sections', []):
                    if sec.get('section') == hit['section']:
                        full = sec.get('text', '')
                        start = hit.get('char_offset', 0)
                        return full[start:start + cfg['chunk_size']]
    return ''

# Pydantic model for incoming query requests("text2text-generation", model=GEN_MODEL, tokenizer=TOKENIZER)

# Pydantic model for incoming query requests
class QueryRequest(BaseModel):
    question: str                       # The user's question (required)
    top_k: Optional[int] = None         # Initial number of chunks to retrieve (optional, falls back to config)
    rerank_top_k: Optional[int] = None  # Number of top results to rerank (optional, falls back to config)

# Register this function as the handler for POST requests to '/query' this function as the handler for POST requests to '/query'
# Clients send HTTP POST to http://<host>:<port>/query to use this endpoint
@app.post("/query")
# Using 'async def' makes this an asynchronous endpoint that won't block on I/O
async def query_law(req: QueryRequest):
    try:
        top_k = req.top_k or cfg['initial_top_k']
        rerank_k = req.rerank_top_k or cfg['rerank_top_k']
        # 1) Embed question
        q_emb = EMBEDDER.encode([req.question], show_progress_bar=False)
        q_vec = np.array(q_emb, dtype='float32')

        # 2) Retrieve
        distances, indices = INDEX.search(q_vec, top_k)
        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            info = META[idx].copy()
            info['distance'] = float(dist)
            hits.append(info)

        # 3) Rerank
        from rag_api import load_text  # import load_text from the correct module  # ensure helper is imported here
        texts = []
        for h in hits:
            texts.append(load_text(h))
        pairs = [[req.question, t] for t in texts]
        scores = CROSS_ENCODER.predict(pairs)
        scored = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:rerank_k]
        top_hits = [h for h, _ in scored]

        # 4) Generate answer) Generate answer
        context = "\n\n".join(
            f"[Law {h['law_id']} §{h['paragraph']} stk.{h['section']}] {load_text(h)}"
            for h in top_hits
        )
        prompt = (
            "You are a Danish legal assistant. Based on the provided excerpts, answer concisely and cite the section.\n\n"
            f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
        )
        out = GEN_PIPE(prompt, max_length=512, do_sample=False)[0]['generated_text']

        return {"answer": out.strip(), "citations": top_hits}

    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host=cfg['host'], port=cfg['port'])