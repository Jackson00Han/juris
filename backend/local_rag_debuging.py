

import os
import glob
import yaml
import json
import faiss
import numpy as np
import requests
import huggingface_hub

# patch for sentence_transformers compatibility
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
from types import SimpleNamespace
from sentence_transformers import SentenceTransformer, CrossEncoder
from smolagents import Agent, Tool
from smolagents.models import Model



# Limit threads to avoid FAISS/OMP issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ─── Configuration & Paths ───────────────────────────────────────────────────
t = os.path.dirname(__file__)
with open(os.path.join(t, "config.yaml"), encoding="utf8") as f:
    cfg = yaml.safe_load(f)

# ─── Initialize embedder & reranker ONCE ──────────────────────────────────────
EMBEDDER = SentenceTransformer(cfg["embed_model"])
CROSS_ENCODER = CrossEncoder(cfg["cross_encoder_model"])

# 1) Load FAISS index & metadata
INDEX = faiss.read_index(cfg["index_path"])
with open(cfg["meta_path"], encoding="utf8") as f:
    META = json.load(f)

# 2) Load law documents into memory
tmp = {}
for path in glob.glob(os.path.join(cfg["out_dir"], "*.json")):
    with open(path, encoding="utf8") as f:
        doc = json.load(f)
    tmp[doc["id"]] = doc
LAW_DOCS = tmp

# 3) Helper: extract the text chunk matching a hit
def load_chunk_text(hit: dict) -> str:
    law = LAW_DOCS.get(hit["law_id"], {})
    for chap in law.get("structured_text", []):
        for para in chap.get("paragraphs", []):
            if para["paragraph"] == hit["paragraph"]:
                for sec in para.get("sections", []):
                    if sec["section"] == hit["section"]:
                        text = sec.get("text", "")
                        start = hit.get("char_offset", 0)
                        return text[start : start + cfg.get("chunk_size", 800)]
    return ""

# 4) Define the retrieval & answer tool
class RAGTool(Tool):
    name = "rag_tool"
    description = "Retrieve and answer questions about Danish law using local Llama3."
    inputs = {"query": {"type": "string", "description": "The user's legal question", "required": True}}
    output_type = "object"

    def forward(self, query: str):
        # 1) Retrieve top-K
        top_k, rerank_k = cfg["initial_top_k"], cfg["rerank_top_k"]
        vec = np.array(EMBEDDER.encode([query]), dtype="float32")
        dists, idxs = INDEX.search(vec, top_k)
        hits = [{**META[i], "distance": float(d)} for d, i in zip(dists[0], idxs[0])]

        # 2) Rerank
        texts = [load_chunk_text(h) for h in hits]
        scores = CROSS_ENCODER.predict([[query, t] for t in texts])
        top_hits = [h for h, _ in sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:rerank_k]]

        # 3) Build context
        context = "\n\n".join(
            f"[Lov {h['law_id']} §{h['paragraph']} stk.{h['section']}] {load_chunk_text(h)}"
            for h in top_hits
        )

        # 4) Call OpenAI-compatible endpoint
        endpoint = os.environ["OPENAI_API_BASE"] + "/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        payload = {
            "model": cfg.get("local_model_name", "llama3.2:1b"),
            "messages": [
                {"role": "system", "content": "You are a Danish legal assistant. Use the excerpts to answer concisely, with citations."},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
            ],
            "max_tokens": cfg.get("max_tokens", 512),
            "temperature": cfg.get("temperature", 0.0),
            "stop": cfg.get("stop", ["\n\n"]),
        }
        resp = requests.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return {"answer": answer, "citations": top_hits}

# 5) Define Ollama-backed Model
class OllamaModel(Model):
    def __init__(self, model_name: str, endpoint: str, max_tokens: int=None, temperature: float=None, stop: list=None):
        super().__init__()
        self.model_name = model_name
        self.endpoint   = endpoint
        self.max_tokens = max_tokens
        self.temperature= temperature
        self.stop       = stop

    def generate(self, messages, stop_sequences=None):
        dict_messages = [{"role": m.role, "content": m.content} for m in messages]
        payload = {
            "model": self.model_name,
            "messages": dict_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": stop_sequences or self.stop,
        }
        resp = requests.post(self.endpoint, json=payload)
        resp.raise_for_status()
        d = resp.json()
        content = d["choices"][0]["message"]["content"].strip()
        u = d.get("usage", {})
        return SimpleNamespace(
            content     = content,
            token_usage = SimpleNamespace(
                input_tokens  = u.get("prompt_tokens", 0),
                output_tokens = u.get("completion_tokens", 0),
                total_tokens  = u.get("total_tokens", 0)
            )
        )

# 6) Instantiate agent without planning
client = OllamaModel(
    model_name=cfg.get("local_model_name","llama3.2:1b"),
    endpoint= os.environ["OPENAI_API_BASE"] + "/chat/completions",
    max_tokens=cfg.get("max_tokens",512),
    temperature=cfg.get("temperature",0.0),
    stop=cfg.get("stop",["\n\n"]),
)
rag_agent = Agent(
    tools=[RAGTool()],
    model=client
)

# 7) Run example
if __name__ == "__main__":
    question = "Hvornår træder bekendtgørelsen i kraft?"
    result = rag_agent.run(question)
    print(json.dumps(result, ensure_ascii=False, indent=2))