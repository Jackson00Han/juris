#!/usr/bin/env python3
"""
Smolagents-based RAG Agent for Danish Law
"""
import os
# Avoid threading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import huggingface_hub
# Patch for sentence-transformers compatibility
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import glob
import json
import yaml
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from smolagents import CodeAgent, Tool, InferenceClientModel

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Load vector index and metadata
INDEX = faiss.read_index(cfg['index_path'])
with open(cfg['meta_path'], encoding='utf8') as f:
    META = json.load(f)

# Initialize models
EMBEDDER = SentenceTransformer(cfg['embed_model'])
CROSS_ENCODER = CrossEncoder(cfg['cross_encoder_model'])
TOKENIZER = AutoTokenizer.from_pretrained(cfg['gen_model'])
GEN_MODEL = AutoModelForSeq2SeqLM.from_pretrained(cfg['gen_model'])
GEN_PIPE = pipeline("text2text-generation", model=GEN_MODEL, tokenizer=TOKENIZER)

# Load full JSON docs
LAW_DOCS = {}
for path in glob.glob(os.path.join(cfg['out_dir'], '*.json')):
    with open(path, encoding='utf8') as f:
        doc = json.load(f)
    LAW_DOCS[doc['id']] = doc

# Helper function to retrieve chunk text
def load_chunk_text(hit: dict) -> str:
    law = LAW_DOCS.get(hit['law_id'], {})
    for chap in law.get('structured_text', []):
        for para in chap.get('paragraphs', []):
            if para.get('paragraph') == hit['paragraph']:
                for sec in para.get('sections', []):
                    if sec.get('section') == hit['section']:
                        start = hit.get('char_offset', 0)
                        text = sec.get('text', '')
                        return text[start:start + cfg['chunk_size']]
    return ''

# Single RAG tool
class RAGTool(Tool):
    name = "rag_tool"
    description = "Retrieve, rerank, and generate answers from Danish law corpus."
    # Define inputs as a dict: parameter name to description/type
    inputs = {
        "query": "The user's legal question",
        "top_k": "Optional[int] initial number of chunks to retrieve",
        "rerank_k": "Optional[int] number of chunks to rerank"
    }

    def run(self, query: str, top_k: int = None, rerank_k: int = None):
        top_k = top_k or cfg['initial_top_k']
        rerank_k = rerank_k or cfg['rerank_top_k']

        # Retrieve
        vec = np.array(EMBEDDER.encode([query]), dtype='float32')
        dists, idxs = INDEX.search(vec, top_k)
        hits = [{**META[idx], 'distance': float(d)} for d, idx in zip(dists[0], idxs[0])]

        # Rerank
        texts = [load_chunk_text(h) for h in hits]
        scores = CROSS_ENCODER.predict([[query, t] for t in texts])
        top_hits = [h for h, _ in sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:rerank_k]]

        # Generate
        context = "\n\n".join(
            f"[Law {h['law_id']} §{h['paragraph']} stk.{h['section']}] {load_chunk_text(h)}"
            for h in top_hits
        )
        prompt = (
            "You are a Danish legal assistant. Answer concisely using the provided excerpts and cite sections.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        answer = GEN_PIPE(prompt, max_length=512, do_sample=False)[0]['generated_text'].strip()
        return {"answer": answer, "citations": top_hits}

# Build and run CodeAgent
agent = CodeAgent(
    tools=[RAGTool()],
    model=InferenceClientModel()
)

# Synchronous invocation of the agent (no HTTP)
if __name__ == "__main__":
    query_text = "Hvad siger § 1 i Retsplejeloven?"
    # Call agent.run with keyword args (not passing a dict)
    result = agent.run(query=query_text, top_k=5, rerank_k=3)
    print(result)
