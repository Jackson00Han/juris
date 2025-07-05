#!/usr/bin/env python3
"""
rag_agent.py

A simple RAG agent for Danish law, using Smolagents.
"""

import os
import glob
import json
import yaml
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from smolagents import CodeAgent, Tool, InferenceClientModel

# ─── Load config ───────────────────────────────────────────────────────────────
with open("config.yaml", encoding="utf8") as f:
    cfg = yaml.safe_load(f)

# ─── Load vector index & metadata ─────────────────────────────────────────────
INDEX = faiss.read_index(cfg["index_path"])
with open(cfg["meta_path"], encoding="utf8") as f:
    META = json.load(f)

# ─── Load full law JSONs ───────────────────────────────────────────────────────
LAW_DOCS = {}
for path in glob.glob(os.path.join(cfg["out_dir"], "*.json")):
    with open(path, encoding="utf8") as f:
        doc = json.load(f)
    LAW_DOCS[doc["id"]] = doc

def load_chunk_text(hit: dict) -> str:
    law = LAW_DOCS.get(hit["law_id"], {})
    for chap in law.get("structured_text", []):
        for para in chap.get("paragraphs", []):
            if para["paragraph"] == hit["paragraph"]:
                for sec in para["sections"]:
                    if sec["section"] == hit["section"]:
                        text = sec.get("text", "")
                        start = hit.get("char_offset", 0)
                        return text[start : start + cfg["chunk_size"]]
    return ""

# ─── Models ────────────────────────────────────────────────────────────────────
EMBEDDER     = SentenceTransformer(cfg["embed_model"])
CROSS_ENCODER= CrossEncoder(cfg["cross_encoder_model"])
TOKENIZER    = AutoTokenizer.from_pretrained(cfg["gen_model"])
GEN_MODEL    = AutoModelForSeq2SeqLM.from_pretrained(cfg["gen_model"])
GEN_PIPE     = pipeline(
    "text2text-generation",
    model=GEN_MODEL,
    tokenizer=TOKENIZER,
)

# ─── RAG Tool Definition ──────────────────────────────────────────────────────
class RAGTool(Tool):
    name        = "rag_tool"
    description = "Retrieve and answer questions about Danish law using RAG."
    inputs = {
        "query":     "The user's legal question",
        "top_k":     "Optional[int]: how many chunks to retrieve initially",
        "rerank_k":  "Optional[int]: how many to rerank & use"
    }

    def run(self, query: str, top_k: int = None, rerank_k: int = None):
        top_k    = top_k    or cfg["initial_top_k"]
        rerank_k = rerank_k or cfg["rerank_top_k"]

        # 1) retrieve
        vec   = np.array(EMBEDDER.encode([query]), dtype="float32")
        dists, idxs = INDEX.search(vec, top_k)
        hits  = [{**META[i], "distance": float(d)} for d, i in zip(dists[0], idxs[0])]

        # 2) rerank
        texts = [load_chunk_text(h) for h in hits]
        scores= CROSS_ENCODER.predict([[query, t] for t in texts])
        top_hits = [
            h for h, _ in sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:rerank_k]
        ]

        # 3) generate answer
        context = "\n\n".join(
            f"[Lov {h['law_id']} §{h['paragraph']} stk.{h['section']}] {load_chunk_text(h)}"
            for h in top_hits
        )
        prompt = (
            "You are a Danish legal assistant. Answer concisely using the excerpts below and cite.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        out = GEN_PIPE(prompt, max_length=512, do_sample=False)[0]["generated_text"].strip()
        return {"answer": out, "citations": top_hits}


model = InferenceClientModel()

rag_agent = CodeAgent(
    tools=[RAGTool()],
    model=model
)

response = rag_agent.run(
    query="Hvornår træder bekendtgørelsen i kraft?",
    top_k=5,
    rerank_k=3
)

print("Answer:", response)