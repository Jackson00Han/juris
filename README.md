# Juris

A full-stack legal document analysis platform leveraging AI to assist with parsing, embedding, and querying legal texts.

## Table of Contents

* [Features](#features)
* [Repository Structure](#repository-structure)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Sprints](#sprints)
* [Running with Docker Compose](#running-with-docker-compose)
* [Development Workflow](#development-workflow)
* [Branching Strategy](#branching-strategy)
* [Contributing](#contributing)
* [License](#license)

## Features

* **Document Parsing**: Extract and preprocess legal documents into text embeddings.
* **Embedding Service**: Compute vector embeddings for semantic search.
* **API Server**: FastAPI backend exposing endpoints for ingestion and querying.
* **Frontend UI**: React-based interface for uploading documents and interacting with the AI search.
* **Dockerized**: Easy setup with Docker and Docker Compose.

## Repository Structure

```plaintext
├── backend             # FastAPI server and ingestion logic
│   ├── app            # Application source code
│   ├── config.py      # Configuration settings
│   ├── requirements.txt
│   └── Dockerfile
├── frontend            # React frontend application
├── first_try           # Legacy prototype and scratch files (archived)
├── docker-compose.yml  # Orchestrates backend, frontend, and supporting services
└── .gitignore          # Ignore patterns (includes backend/.venv)
```

## Prerequisites

* Docker & Docker Compose
* Git 2.x
* Node.js & npm (for local frontend dev)
* Python 3.10+

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Jackson00Han/juris.git
   cd juris
   ```

2. **Create a Python virtual environment** (optional if using Docker)

   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies** (optional if using Docker)

   ```bash
   cd ../frontend
   npm install
   ```

## Configuration

1. Copy `.env` templates and fill in API keys or database URLs:

   ```bash
   cp backend/.env.example backend/.env
   cp first_try/.env.example first_try/.env
   ```
2. Update environment variables as needed.

## Sprints

We plan development in timeboxed sprints:

* **Sprint 0: Project Setup & Scaffolding**

  * Initialize repo, branches (main/dev)
  * Basic README, .gitignore, CI pipeline stub
  * Docker Compose skeleton

* **Sprint 1: Document Ingestion & Embedding**

  * Implement FastAPI endpoints for PDF/text upload
  * Parse and preprocess documents
  * Integrate embedding model and vector store

* **Sprint 2: Semantic Search API & Frontend Prototype**

  * Build search/query endpoints
  * Create React UI for upload & query
  * Display search results and embedding visualizations

* **Sprint 3: Authentication & User Management**

  * Add auth layer (JWT/session)
  * User registration and login flows

* **Sprint 4: Testing & Performance Optimization**

  * Write end-to-end tests (pytest, Jest)
  * Optimize embedding pipeline and caching

* **Sprint 5: Deployment & Monitoring**

  * Configure Kubernetes/Cloud deployment
  * Set up logging, metrics, and alerts

## Running with Docker Compose

Bring up the full stack:

```bash
docker-compose up --build
```

* **Backend** at `http://localhost:8000`
* **Frontend** at `http://localhost:3000`

## Development Workflow

1. **Backend**

   ```bash
   cd backend
   source .venv/bin/activate
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
2. **Frontend**

   ```bash
   cd frontend
   npm start
   ```

## Branching Strategy

* **main**: Protected. Holds the latest stable releases. Merges via pull requests only.
* **dev**: Integration branch for active development. Feature branches branch off `dev` and merge back via PRs.

**Feature Branch Naming**: `feature/<short-summary>`

Example:

```bash
git checkout dev
git checkout -b feature/add-search-endpoint
# implement...
git push origin feature/add-search-endpoint
```

Open a PR into `dev` for review. After `dev` is stable, merge `dev` into `main` via PR.

## Contributing

1. Fork the repository.
2. Create a feature branch from `dev`.
3. Commit changes with clear messages.
4. Push to your fork and open a PR targeting `dev`.
5. Address review comments.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
