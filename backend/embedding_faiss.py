import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK punkt model is downloaded (uncomment if needed)
# nltk.download("punkt")

# Root directory for storing FAISS indices
INDEX_ROOT = os.path.join(os.path.dirname(__file__), "..", "faiss_indexes")

# Embedding model must match the one used by the application
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")


def chunk_text_with_overlap(
    full_text: str,
    max_chars: int = 500,
    overlap_chars: int = 50
) -> list[str]:
    """
    Split full_text into chunks of at most max_chars characters, ensuring:
      - No sentence is split across chunks: first use sentence tokenization,
        then concatenate sentences until the limit is reached.
      - Adjacent chunks overlap by overlap_chars characters for continuity.

    Args:
        full_text (str): The complete text to be chunked.
        max_chars (int): Maximum characters per chunk.
        overlap_chars (int): Number of overlapping characters between adjacent chunks.

    Returns:
        list[str]: List of text chunks.
    """
    sentences = sent_tokenize(full_text)
    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}" if current else sent
        else:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars < len(current) else current
            current = f"{tail} {sent}"

            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    sub = sent[i : i + max_chars]
                    chunks.append(sub)
                current = ""

    if current:
        chunks.append(current)

    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of text chunks into embeddings.

    Args:
        texts (list[str]): List of text strings to embed.

    Returns:
        np.ndarray: A float32 numpy array of shape (len(texts), embedding_dim).
    """
    embeddings = _model.encode(texts, convert_to_numpy=True)
    return np.array(embeddings, dtype="float32")


def save_faiss_index(doc_id: str, texts: list[str], embeddings: np.ndarray):
    """
    Save FAISS index and corresponding text chunks to disk.

    Args:
        doc_id (str): Unique identifier as subdirectory name.
        texts (list[str]): List of text chunks.
        embeddings (np.ndarray): Float32 array of shape (N, D).

    The function creates INDEX_ROOT/doc_id/ and saves:
      1) texts.pkl  – Pickled list of text chunks.
      2) index.faiss – FAISS index of embeddings.
    """
    user_folder = os.path.join(INDEX_ROOT, doc_id)
    os.makedirs(user_folder, exist_ok=True)

    with open(os.path.join(user_folder, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(user_folder, "index.faiss"))


def load_faiss_index(doc_id: str):
    """
    Load a saved FAISS index and its text chunks.

    Args:
        doc_id (str): Unique identifier under which index and texts are stored.

    Returns:
        tuple: (index, texts) where index is a FAISS Index object and
               texts is a list of text chunks.

    Raises:
        FileNotFoundError: If index.faiss or texts.pkl is missing.
    """
    user_folder = os.path.join(INDEX_ROOT, doc_id)
    idx_file = os.path.join(user_folder, "index.faiss")
    txt_file = os.path.join(user_folder, "texts.pkl")

    if not (os.path.exists(idx_file) and os.path.exists(txt_file)):
        raise FileNotFoundError(f"Index not found for {doc_id}")

    index = faiss.read_index(idx_file)
    with open(txt_file, "rb") as f:
        texts = pickle.load(f)
    return index, texts


def search_faiss(index, texts: list[str], query: str, top_k: int) -> list[str]:
    """
    Perform a similarity search on a FAISS index given a query string.

    Args:
        index (faiss.Index): FAISS index object containing embeddings.
        texts (list[str]): List of text chunks corresponding to index rows.
        query (str): The query string to embed and search for.
        top_k (int): Number of top results to retrieve.

    Returns:
        list[str]: List of the top_k most similar text chunks.
    """
    q_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec, top_k)
    results = []
    for idx in I[0]:
        if idx < len(texts):
            results.append(texts[idx])
    return results
