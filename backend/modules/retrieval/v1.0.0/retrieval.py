from embedding_faiss import load_faiss_index, search_faiss
from transformers import pipeline

def run(inputs: dict) -> dict:
    doc_id = inputs["doc_id"]
    question = inputs["question"]
    top_k = inputs.get("top_k", 3)

    index, texts = load_faiss_index(doc_id)
    contexts = search_faiss(index, texts, question, top_k)
    context = "\n\n".join(contexts)

    llm = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    raw = llm(prompt, max_length=200)[0]["generated_text"]
    answer = raw.split("Answer:")[-1].strip()

    return {"answer": answer, "context": context}