# backend/app/main.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import logging
import uvicorn
import fitz  # PyMuPDF
from transformers import pipeline
import torch
from config import settings

from embedding_faiss import (
    chunk_text_with_overlap,
    embed_texts,
    save_faiss_index,
    load_faiss_index,
    search_faiss,
)

if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("juris_backend")

logger.info(f"Loading embedding model: {settings.embed_model} (CPU only)")
# We still need a model instance here for encoding queries in /ask
embed_model = sentence_transformers.SentenceTransformer(
    settings.embed_model, device="cpu"
)
logger.info(f"Loading LLM model: {settings.llm_model} (CPU only)")
gpt_model = pipeline("text2text-generation", model=settings.llm_model, device=-1)

app = FastAPI(
    title="Juris RAG API",
    version="0.1.0",
    description="API for document-based RAG question answering and legal agent orchestration"
)


@app.get("/")
def root():
    return {"message": "Welcome to Juris RAG API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


class AnswerResponse(BaseModel):
    answer: str
    context: str


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "\n".join(page.get_text() for page in doc)

        # Use sentence-based chunking with overlap
        chunks = chunk_text_with_overlap(full_text, max_chars=500, overlap_chars=50)
        embeddings = embed_texts(chunks)
        save_faiss_index(doc_id, chunks, embeddings)

        return {"status": "success", "num_chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error in /upload: {e}")
        raise HTTPException(status_code=500, detail="Upload or indexing failed.")


@app.post("/ask", response_model=AnswerResponse)
async def ask(
    doc_id: str = Form(...),
    question: str = Form(...),
    top_k: int = Form(3)
):
    try:
        index, texts = load_faiss_index(doc_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index not found for the given doc_id.")
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        raise HTTPException(status_code=500, detail="Failed to load index.")

    try:
        contexts = search_faiss(index, texts, question, top_k)
        context = "\n\n".join(contexts)
    except Exception as e:
        logger.error(f"Error in FAISS search: {e}")
        raise HTTPException(status_code=500, detail="Search failed.")

    try:
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
        output = gpt_model(prompt, max_length=settings.max_length)
        raw = output[0].get("generated_text", "")
        answer = raw.split("Answer:")[-1].strip()
    except Exception as e:
        logger.error(f"Error in LLM generation: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed.")

    return AnswerResponse(answer=answer, context=context)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port, reload=True)
