import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

import os

# 限制 OpenMP/BLAS 线程
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import faiss
# Faiss 也控制在单线程
faiss.omp_set_num_threads(1)

import torch
torch.set_num_threads(1)



# 如果第一次运行，需要下载 punkt 模型
# nltk.download("punkt")

# 配置
LAWS_JSON_DIR = "laws_json"  # 之前保存 JSON 的目录
INDEX_ROOT = "faiss_indexes"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHARS = 500
OVERLAP = 100

# 初始化模型
_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

def chunk_text_with_overlap(full_text: str, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP) -> list[str]:
    sentences = sent_tokenize(full_text)
    chunks, current = [], ""
    for sent in sentences:
        sent = sent.strip()
        if not sent: continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}" if current else sent
        else:
            chunks.append(current)
            # 为下一个 chunk 保留尾部 overlap
            tail = current[-overlap_chars:] if overlap_chars < len(current) else current
            current = f"{tail} {sent}"
            # 如果单句本身超长，强制分片
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i:i+max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks

def embed_texts(texts: list[str]) -> np.ndarray:
    embs = _model.encode(texts, convert_to_numpy=True)
    return np.array(embs, dtype="float32")

def save_faiss_index(doc_id: str, chunks: list[dict], embeddings: np.ndarray):
    """
    chunks: list of dicts, each has keys 'paragraph','section','sub_idx','text'
    embeddings: np.ndarray of shape (N, D)
    """
    outdir = os.path.join(INDEX_ROOT, doc_id)
    os.makedirs(outdir, exist_ok=True)
    # 1) 保存所有 chunk 元数据
    with open(os.path.join(outdir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    # 2) 建立 FAISS 并保存
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(outdir, "index.faiss"))
    print(f"✅ {doc_id}: {len(chunks)} chunks, index saved to {outdir}")

def build_all_indices():
    # 遍历所有 JSON
    for fname in os.listdir(LAWS_JSON_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(LAWS_JSON_DIR, fname)
        with open(path, encoding="utf-8") as f:
            law = json.load(f)
        doc_id = law.get("id")
        structured = law.get("structured_text", [])
        chunks = []
        # 扁平化
        for para in structured:
            p = para.get("paragraph","")
            for sec in para.get("sections", []):
                s = sec.get("section","")
                text = sec.get("text","").strip()
                if not text:
                    continue
                # 短文本直接一个 chunk
                if len(text) <= MAX_CHARS:
                    chunks.append({"paragraph": p, "section": s, "sub_idx": 0, "text": text})
                else:
                    subs = chunk_text_with_overlap(text)
                    for i, sub in enumerate(subs):
                        chunks.append({"paragraph": p, "section": s, "sub_idx": i, "text": sub})
        if not chunks:
            print(f"⚠️ {doc_id} 没有可切分的文本，跳过")
            continue
        # 批量 embedding
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        # 存索引
        save_faiss_index(doc_id, chunks, embeddings)

if __name__ == "__main__":
    build_all_indices()
