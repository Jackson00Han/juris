# backend/app/main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import logging
import uvicorn
import fitz  # PyMuPDF for PDF text extraction
import torch
from transformers import pipeline
from config import settings

# Local implementations: retrieval & embedding, chunking, FAISS interface
from embedding_faiss import (
    chunk_text_with_overlap,
    embed_texts,
    save_faiss_index,
    load_faiss_index,
    search_faiss,
)
# Import the appropriate versioned run() functions from modules/
from modules.retrieval.v1_0_0.retrieval import run as retrieval_run
from modules.template.v1_0_0.template import run as template_run
from modules.logic.v1_0_0.logic import run as logic_run

# Ensure torch.get_default_device exists (for older torch builds)
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("juris_backend")

logger.info(f"Loading LLM model: {settings.llm_model} (CPU only)")
gpt_model = pipeline("text2text-generation", model=settings.llm_model, device=-1)

app = FastAPI(
    title="Juris RAG API",
    version="0.1.0",
    description="Unified agent endpoint for QA and contract drafting",
)

class AgentResponse(BaseModel):
    task_type: str

    # The following fields are optional; they may be null depending on the branch
    answer:         str | None = None
    context:        str | None = None
    generated_text: str | None = None
    logic_results:  list[dict] | None = None
    overall_passed: bool | None = None

# ——— Upload & index endpoint unchanged from before ———
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

        # Chunking + embedding + saving to FAISS
        chunks = chunk_text_with_overlap(full_text, max_chars=500, overlap_chars=50)
        embeddings = embed_texts(chunks)
        save_faiss_index(doc_id, chunks, embeddings)

        return {"status": "success", "num_chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error in /upload: {e}")
        raise HTTPException(status_code=500, detail="Upload or indexing failed.")

# ——— Unified Agent endpoint ———
@app.post("/run_agent", response_model=AgentResponse)
async def run_agent(
    # “task_type” determines which modules to run; example values: ["qa", "draft_contract"]
    task_type: str,
    # Fields for retrieval (applicable to QA or Draft modes; can be empty if not needed)
    doc_id: str | None = None,
    question: str | None = None,
    top_k: int = 3,

    # Fields for template rendering (required for Draft mode; can be empty in QA mode)
    template_id: str | None = None,
    template_data: dict | None = None,

    # Fields for compliance (Logic) module (same as above)
    rules: list[dict] | None = None
):
    result = {
        "task_type": task_type,
        "answer": None,
        "context": None,
        "generated_text": None,
        "logic_results": None,
        "overall_passed": None
    }

    # ——— ① QA Mode: only run retrieval and return answer + context ———
    if task_type == "qa":
        if not doc_id or not question:
            raise HTTPException(status_code=400, detail="doc_id and question are required for QA.")
        inputs = {"doc_id": doc_id, "question": question, "top_k": top_k}
        outputs = retrieval_run(inputs)
        result["answer"]  = outputs["answer"]
        result["context"] = outputs["context"]
        return result

    # ——— ② Draft Mode (Contract Drafting): optionally retrieve → template rendering → compliance ———
    if task_type == "draft_contract":
        # 1) If doc_id and question are provided, treat as “retrieve reference clauses”; otherwise skip retrieval
        if doc_id and question:
            inputs = {"doc_id": doc_id, "question": question, "top_k": top_k}
            outputs = retrieval_run(inputs)
            result["answer"]  = outputs["answer"]
            result["context"] = outputs["context"]
        # 2) Template rendering: template_id & template_data must exist
        if not template_id or not template_data:
            raise HTTPException(status_code=400, detail="template_id and template_data are required for drafting.")
        tmpl_inputs = {"template_id": template_id, "data": template_data}
        tmpl_outputs = template_run(tmpl_inputs)
        result["generated_text"] = tmpl_outputs["generated_text"]

        # 3) Compliance Module: only run if rules parameter is provided; otherwise skip
        if rules:
            logic_inputs = {"rules": rules, "draft_text": result["generated_text"]}
            logic_outputs = logic_run(logic_inputs)
            result["logic_results"]  = logic_outputs["results"]
            result["overall_passed"] = logic_outputs["overall_passed"]

        return result

    # ——— ③ Unsupported task_type → Error ———
    raise HTTPException(status_code=400, detail=f"Unsupported task_type: {task_type}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port, reload=True)
