# Legal Assistant

This repository contains a Flask-based Legal Assistant that accepts PDF uploads and answers questions based on the document content.

## Prerequisites

* Python 3.10+ installed
* Git installed

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-org/juris.git
   cd juris
   ```
2. **Create and activate a Conda environment**:

   ```bash
   conda create -n legal-assistant python=3.10
   conda activate legal-assistant
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## requirements.txt

```text
Flask==2.2.5
PyMuPDF==1.22.5
sentence-transformers==2.2.2
transformers==4.31.0
torch>=1.12.0
gunicorn==20.1.0
```

## Running Locally

```bash
export PORT=5001  # or choose any other free port
python app.py
```

Then open your browser to [http://127.0.0.1:5001](http://127.0.0.1:5001).

## Next Steps

* [ ] Containerize with Docker
* [ ] Setup CI/CD pipeline
* [ ] Implement error handling and logging
