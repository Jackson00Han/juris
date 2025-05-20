# Workaround: ensure cached_download is available at the top level of huggingface_hub
import huggingface_hub
from huggingface_hub import hf_hub_download as cached_download
setattr(huggingface_hub, "cached_download", cached_download)

from flask import Flask, request, jsonify
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

app = Flask(__name__)

# Health-check endpoint

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>LB Assistant</h1>
    <form action="/ask" method="post" enctype="multipart/form-data">
      <label>PDF: <input type="file" name="file" required/></label><br/>
      <label>Question: <input type="text" name="question" size="60" required/></label><br/>
      <button type="submit">Ask</button>
    </form>
    '''


# Load models once
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt_model = pipeline('text2text-generation', model='google/flan-t5-small')

# Helpers
def extract_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    return ''.join(page.get_text() for page in doc)


def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

@app.route('/ask', methods=['POST'])
def ask():
    file = request.files.get('file')
    question = request.form.get('question')
    if not file or not question:
        return jsonify(error='Provide a PDF file and a question'), 400

    text = extract_text(file.read())
    chunks = chunk_text(text)

    # Compute embeddings
    embeddings = embed_model.encode(chunks)
    q_embed = embed_model.encode([question])[0]

    # Simple similarity (dot product)
    scores = [sum(e * q for e, q in zip(vec, q_embed)) for vec in embeddings]
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    context = '\n\n'.join(chunks[i] for i in top_idxs)

    # Generate answer
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    out = gpt_model(prompt, max_length=200)[0]['generated_text']
    answer = out.split('Answer:')[-1].strip()

    return jsonify(answer=answer)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)