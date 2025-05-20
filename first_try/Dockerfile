FROM python:3.10-slim

# 1. Install system build tools once
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy only the requirements and install (caching layers)
COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt && \
    python - <<EOF
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Pre-cache embeddings model
SentenceTransformer('all-MiniLM-L6-v2')
# Pre-cache Flan-T5 model and tokenizer
T5Tokenizer.from_pretrained('google/flan-t5-small')
T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
EOF

# 3. Copy application code
COPY . .

# 4. Expose and set up environment
EXPOSE 5001
ENV PYTHONUNBUFFERED=1

# 5. Launch with Gunicorn, preloading app and generous timeout
CMD ["gunicorn","app:app","--bind","0.0.0.0:5001","--workers","2","--preload","--timeout","600"]
