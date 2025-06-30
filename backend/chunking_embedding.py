#!/usr/bin/env python3
import os
# Limit threading to avoid segmentation faults at exit
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import glob
import argparse
from typing import List, Dict, Tuple

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys

# ─── CONFIG ────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE        = 800        # characters per chunk
CHUNK_OVERLAP     = 200        # characters overlap
BATCH_SIZE        = 32         # how many chunks to embed at once

# ─── GLOBAL MODEL PLACEHOLDER ─────────────────────────────────────────────────
MODEL = None

# ─── UTILITIES ────────────────────────────────────────────────────────────────

def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP
              ) -> List[Tuple[str,int]]:
    """
    Split 'text' into overlapping chunks of up to chunk_size chars.
    Returns list of (chunk_text, start_char_index).
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append((text[start:end], start))
        start += (chunk_size - overlap)
    return chunks


def load_documents(data_dir: str) -> List[Dict]:
    """
    Read all JSON files in data_dir and flatten into units:
    each with 'text' and metadata (law_id, chapter, paragraph, section).
    """
    units = []
    for path in glob.glob(os.path.join(data_dir, "*.json")):
        data = json.load(open(path, encoding="utf8"))
        law_id = data.get("id")
        for chap in data.get("structured_text", []):
            chap_num = chap.get("chapter", "")
            for para in chap.get("paragraphs", []):
                para_num = para.get("paragraph", "")
                for sec in para.get("sections", []):
                    sec_num = sec.get("section", "")
                    text = sec.get("text", "").strip()
                    if not text:
                        continue
                    units.append({
                        "law_id": law_id,
                        "chapter": chap_num,
                        "paragraph": para_num,
                        "section": sec_num,
                        "text": text
                    })
    return units


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts via all-MiniLM-L6-v2.
    Returns a list of embeddings.
    """
    global MODEL
    # MODEL should be initialized before calling
    embs = MODEL.encode(texts, batch_size=len(texts), show_progress_bar=False)
    return embs.tolist()

# ─── MAIN INGEST ──────────────────────────────────────────────────────────────

def main():
    global MODEL
    parser = argparse.ArgumentParser(
        description="Chunk & embed Danish laws into a FAISS index"
    )
    parser.add_argument("--data-dir", "-d", default="laws_json",
                        help="Directory containing law JSON files")
    parser.add_argument("--index-path", "-i", default="laws_mini.faiss",
                        help="Output path for FAISS index")
    parser.add_argument("--meta-path", "-m", default="laws_mini_meta.json",
                        help="Output path for chunk metadata JSON")
    args = parser.parse_args()

    # Initialize model locally to avoid threading issues
    print(f"Loading embedder model: {EMBED_MODEL_NAME}...")
    MODEL = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Loading documents from {args.data_dir}…")
    units = load_documents(args.data_dir)
    print(f"  → {len(units):,} sections loaded")

    chunks   = []
    all_meta = []
    for u in units:
        for text_chunk, offset in chunk_text(u["text"]):
            chunks.append(text_chunk)
            all_meta.append({
                "law_id":      u["law_id"],
                "chapter":     u["chapter"],
                "paragraph":   u["paragraph"],
                "section":     u["section"],
                "char_offset": offset
            })

    total = len(chunks)
    print(f"Total chunks: {total:,}")

    all_embeddings = []
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding batches"):
        batch_texts = chunks[i:i+BATCH_SIZE]
        batch_embs  = embed_batch(batch_texts)
        all_embeddings.extend(batch_embs)

    mat = np.array(all_embeddings, dtype="float32")
    dim = mat.shape[1]
    print(f"Embedding matrix shape: {mat.shape} (dim={dim})")

    index = faiss.IndexFlatL2(dim)
    index.add(mat)
    faiss.write_index(index, args.index_path)
    print(f"FAISS index written to {args.index_path}")

    with open(args.meta_path, "w", encoding="utf8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(f"Metadata JSON written to {args.meta_path}")

    # Explicitly exit to bypass potential threading teardown issues
    sys.exit(0)

if __name__ == "__main__":
    main()
