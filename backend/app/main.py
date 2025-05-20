# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import uvicorn
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Optional
import torch
from config import settings

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("juris_backend")

# Load models once at startup
logger.info(f"Loading embedding model: {settings.embed_model}")
embed_model = SentenceTransformer(settings.embed_model)
logger.info(f"Loading LLM model: {settings.llm_model}")
gpt_model = pipeline(
    "text2text-generation",
    model=settings.llm_model,
    device=settings.device
)

# FastAPI app
app = FastAPI(
    title="Juris RAG API",
    version="0.1.0",
    description="API for document-based RAG question answering and legal agent orchestration"
)

@app.get("/")
def root():
    return {"message": "Welcome to Juris RAG API"}



class AnswerResponse(BaseModel):
    answer: str
    context: Optional[str]

@app.get("/health")
def health_check():
    return {"status": "ok"}

async def extract_text(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        raise HTTPException(status_code=500, detail="Text extraction failed.")

async def chunk_text(text: str, size: int = 500) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]

@app.post("/ask", response_model=AnswerResponse)
async def ask(
    file: UploadFile = File(...),
    question: str = Form(...),
    top_k: int = Form(3)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    logger.info("Extracting text from PDF...")
    text = await extract_text(pdf_bytes)

    logger.info("Chunking text...")
    chunks = await chunk_text(text)

    logger.info("Encoding chunks...")
    embeddings = embed_model.encode(chunks, convert_to_tensor=True)
    q_embedding = embed_model.encode([question], convert_to_tensor=True)

    logger.info("Computing similarities...")
    sims = torch.nn.functional.cosine_similarity(embeddings, q_embedding)
    allowed_k = min(top_k, len(chunks))
    top_idxs = torch.topk(sims, k=allowed_k).indices.tolist()
    context = "\n\n".join(chunks[i] for i in top_idxs)

    logger.info("Generating answer via LLM...")
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    output = gpt_model(prompt, max_length=settings.max_length)
    raw = output[0].get("generated_text", "")
    answer = raw.split("Answer:")[-1].strip()

    return AnswerResponse(answer=answer, context=context)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port, reload=True)