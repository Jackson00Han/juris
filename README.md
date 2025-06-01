# Juris

A full-stack legal document analysis platform leveraging AI to assist with parsing, embedding, querying legal texts, and—most importantly—enabling **Modular Legal Agents** for collaborative, version-controlled workflows.

## Table of Contents

- [Features](#features)  
- [Modular Agent MVP 1 Workflow & Architecture](#modular-agent-mvp-1-workflow--architecture)  
  - [Overview](#overview)  
  - [High-Level Phases](#high-level-phases)  
- [Repository Structure](#repository-structure)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Sprints](#sprints)  
- [Running with Docker Compose](#running-with-docker-compose)  
- [Development Workflow](#development-workflow)  
- [Branching Strategy](#branching-strategy)  
- [License](#license)

---

## Features

- **Document Parsing**: Extract and preprocess legal documents (PDFs, Word, HTML) into plain text.  
- **Embedding Service**: Compute vector embeddings (via Sentence-Transformers) for semantic search.  
- **Vector Retrieval API**: FAISS-powered local or cloud-based vector store for fast similarity lookups.  
- **Modular Legal Agent Framework**:  
  - *Version-controlled modules* (Retrieval, Drafting/Template, Logic/Compliance, etc.)  
  - *Composable flows*: drag-and-drop or JSON-configure a pipeline of “blocks” to build custom Agents.  
  - *Collaborative & audit-ready*: each module is stored in Git (YAML + code), all changes are versioned, enabling “legal wiki/GitHub” for law.  
- **API Server**: FastAPI backend exposing endpoints for ingestion, retrieval, and flow execution.  
- **Frontend UI**: React-based interface for uploading documents, assembling Agents, and viewing results.  
- **Dockerized**: One-command setup with Docker and Docker Compose for local development or self-hosting.

---

## Modular Agent MVP 1 Workflow & Architecture

### Overview

Juris’s next milestone (MVP 1) is to layer a **Modular Agent Engine** on top of our existing RAG (Retrieval-Augmented Generation) foundation. Instead of one monolithic “legal bot,” we want a factory of *mini-agents* (modules) that can be:

1. **Defined & maintained** via Git (YAML metadata + Python code).  
2. **Composed** into an ordered “Flow” (pipeline) that runs step-by-step.  
3. **Versioned** so that “Retrieval v1.2.0” or “Drafting v0.5.3” can be rolled back or branched.  
4. **Collaborated on** by multiple team members or external experts (legal community).

Each “agent” is just a sequence of modules—e.g.:

- **Retrieval → Template → Compliance**  
- **DataInput → Logic → Drafting**  

By the end of MVP 1, you should be able to:

1. Upload a PDF (or fill in data fields).  
2. In the UI, pick from available modules and set their versions.  
3. Configure each module’s parameters (e.g. “top_k = 5” for Retrieval, “include_deposit_clause = true” for Drafting).  
4. Save and version your Flow as `agent_config.json`.  
5. Run the Flow: Backend executes each module in sequence, passing outputs forward.  
6. Display final document/answer back in the browser.

---

### High-Level Phases

#### Phase 0: Audit & Requirements (1 week)

- **Architecture Review**  
  - Examine current RAG components:  
    - **Scraper**: how PDFs and legal sites are ingested.  
    - **Parser**: text extraction (PyMuPDF).  
    - **Embedding**: Sentence-Transformers.  
    - **Vector Store**: FAISS (local) or a hosted vector DB (later).  
    - **Existing API**: Flask or FastAPI endpoints.  
    - **Frontend**: Basic upload + question UI.  
  - Document each component’s responsibilities, language/framework, and interfaces (input/output).
  - Identify any **tech debt** (e.g. “Retrieval takes >3 seconds for large corpora,” “Parser misses footnotes,” “Hardwired Flask routes with no plugin”).

- **Use-Case & Module Definition**  
  - With product/legal stakeholders, agree on **3–4 core module types** for MVP 1:  
    1. **Retrieval Module** (RAG search)—Given a question, return top-k legal text snippets.  
    2. **Template Module** (Drafting)—Given data fields or context + question, render a document from a predefined template.  
    3. **Logic/Compliance Module**—Given a draft or data, apply “If…then…” business rules or LLM prompts to check for risk/requirements.  
  - For each module, sketch a simple **JSON I/O contract** (field names + types):
    - Retrieval: `{ "question": string } → { "contexts": [string] }`  
    - Template: `{ "data": object, "template_id": string } → { "generated_text": string }`  
    - Logic: `{ "input_text": string, "config": object } → { "decision": object }`  

- **Deliverables**  
  1. **System Architecture Diagram** (e.g. Draw.io) showing existing RAG path plus where “modules/” will plug in.  
  2. **Functional Requirements Doc** detailing each module’s inputs, outputs, and sample use cases.

---

#### Phase 1: Modular Framework Design (2 weeks)

1. **Define Module Contract**  
   - Create a uniform **YAML schema** for module metadata (`module.yaml`):  
     ```yaml
     name: string           # e.g. “retrieval”, “template”, “compliance”
     version: semver        # e.g. “1.0.0”, “0.2.5”
     description: string    # short description
     inputs:
       - name: field1
         type: string       # or “object” for nested JSON
       - name: field2
         type: integer
     outputs:
       - name: result
         type: string
     config:
       type: object         # free-form config JSON; validated at runtime
     ```
   - Require each module to implement a Python class with a `process(inputs: dict, config: dict) -> dict` method.

2. **Module Registry & Versioning**  
   - In the backend repo, create a top-level `modules/` folder. Under it, subfolders by module name, then by version:  
     ```
     backend/
       modules/
         retrieval/
           v1.0.0/
             module.yaml
             implementation.py  # defines class RetrievalModule
           v1.1.0/
         template/
           v0.1.0/
             module.yaml
             implementation.py
         compliance/
           v0.1.0/
             module.yaml
             implementation.py
     ```
   - Build a **Module Loader** which at startup scans `modules/`, reads each `module.yaml`, dynamically imports `implementation.py`, and registers the module under `{name}@{version}`.

3. **Orchestration Engine (Flow Runner)**  
   - Define a JSON format for Agent Flows (`agent_config.json`):  
     ```json
     {
       "agent_name": "Lease-Agreement-Flow",
       "steps": [
         {
           "module": "retrieval",
           "version": "1.1.0",
           "config": { "top_k": 5 }
         },
         {
           "module": "template",
           "version": "0.1.0",
           "config": { "template_id": "basic-lease-v1" }
         },
         {
           "module": "compliance",
           "version": "0.1.0",
           "config": { "check_types": ["GDPR", "local-ordinance"] }
         }
       ]
     }
     ```
   - The **Flow Runner** in Python will:
     1. Load each specified module via the Module Loader.  
     2. For step 1: call `Mod1.process(inputs= <e.g. question/previous result>, config=step1.config)`.  
     3. Take output of step 1, feed it as input to step 2, and so on.  
     4. Return the final output as a single JSON to the client.

- **Deliverables**  
  1. **Module Contract Doc** (`DEVELOPER_GUIDE.md` excerpt with YAML schema examples).  
  2. **Module Loader Code** (`backend/modules/loader.py`) that returns a registry map:  
     ```python
     {
       "retrieval@1.0.0": RetrievalModuleClass,
       "template@0.1.0": TemplateModuleClass,
       "compliance@0.1.0": ComplianceModuleClass
     }
     ```
  3. **Flow Runner Skeleton** (`backend/core/flow_runner.py`) that executes a given `agent_config.json`.

---

#### Phase 2: Core Module Implementation & Integration (3–4 weeks)

1. **Develop Individual Modules**  
   - **Retrieval Module**  
     - Wrap existing RAG logic: input `{ "question": str }` → output `{ "contexts": [str] }`.  
     - Example code sketch in `modules/retrieval/v1.0.0/implementation.py`:  
       ```python
       from sentence_transformers import SentenceTransformer
       import faiss
       class RetrievalModule:
           def __init__(self):
               self.model = SentenceTransformer('all-MiniLM-L6-v2')
               self.index, self.texts = load_faiss_index()
           def process(self, inputs, config):
               q_embed = self.model.encode([inputs["question"]])[0]
               D, I = self.index.search(q_embed.reshape(1, -1).astype('float32'), config.get("top_k", 5))
               contexts = [ self.texts[i] for i in I[0] ]
               return {"contexts": contexts}
       ```  
     - Write unit tests (`tests/test_retrieval.py`) that mock a small FAISS index and verify correctness.

   - **Template Module**  
     - Use a templating engine (e.g. Jinja2) to render documents.  
     - Example in `modules/template/v0.1.0/implementation.py`:  
       ```python
       from jinja2 import Environment, FileSystemLoader
       class TemplateModule:
           def __init__(self):
               self.env = Environment(loader=FileSystemLoader('modules/template/v0.1.0/templates'))
           def process(self, inputs, config):
               tmpl = self.env.get_template(f"{config['template_id']}.html")
               generated = tmpl.render(**inputs["data"])
               return {"generated_text": generated}
       ```  
     - Include a sample template file (`modules/template/v0.1.0/templates/basic_contract.html`) with placeholders like `{{ partyA }}`.  
     - Unit test ensures placeholders are correctly replaced.

   - **Compliance Module**  
     - Combine simple rule engines with an LLM-based diagnose.  
     - Example in `modules/compliance/v0.1.0/implementation.py`:  
       ```python
       from transformers import pipeline
       class ComplianceModule:
           def __init__(self):
               self.llm = pipeline('text2text-generation', model='google/flan-t5-small')
           def process(self, inputs, config):
               draft = inputs["generated_text"]
               prompt = f\"Check the following draft for {', '.join(config['check_types'])}: \\n\\n{draft}\"
               result = self.llm(prompt, max_length=200)[0]["generated_text"]
               return {"compliance_report": result}
       ```  
     - Unit test provides a dummy contract text and asserts that “GDPR” checks appear in the output.

2. **End-to-End Flow Testing**  
   - Prepare 2–3 real or sample legal documents: e.g.  
     - A PDF of a lease regulation.  
     - A template JSON for contract fields.  
   - Write CLI or pytest scripts (`tests/test_end_to_end.py`) that:  
     1. Load `agent_config_lease.json` (retrieval → template → compliance).  
     2. Simulate user uploading a PDF → parse text → run Flow Runner.  
     3. Assert that final output contains expected keywords or structure.

3. **Front-End Prototype**  
   - In `frontend/`, add a new React page (e.g. `AgentBuilder.jsx`):  
     - **Module Palette** (left side): dropdowns or draggable cards listing modules by name/version + short description.  
     - **Canvas** (center): a vertical list or drag-target where selected modules drop in order.  
     - **Module Config Panel** (right): when a module is clicked, show inputs for `config` keys (top_k, template_id, check_types).  
     - **Run Button** (bottom): triggers a POST to `/api/flow/run` with the assembled JSON.  
     - **Output Panel**: area below or in a modal that displays the returned JSON (contexts, generated_text, compliance_report).  
   - Ensure the front end can fetch the module registry from a new endpoint (`GET /api/modules`) to populate the palette dynamically.

- **Deliverables**  
  1. **Complete implementation** of `RetrievalModule`, `TemplateModule`, `ComplianceModule` under `backend/modules/...`.  
  2. **Unit tests** for each module in `tests/`.  
  3. **Flow runner tests** demonstrating 1–2 full scenarios.  
  4. **React prototype** with a working “Agent Builder” page that invokes the backend Flow Runner and displays results.

---

#### Phase 3: Documentation, Testing & Internal Launch (1–2 weeks)

1. **Finalize README & Developer Guide**  
   - Update **README.md** (root) to include:  
     - How to define a new module (folder structure, `module.yaml` format, coding guidelines).  
     - How to assemble an Agent Flow (JSON schema, sample `agent_config.json`).  
     - Example end-to-end invocation steps.  
   - Create a **DEVELOPER_GUIDE.md** under `backend/` with deeper instructions on:  
     1. Module versioning best practices (semantic versioning).  
     2. Testing guidelines (pytest patterns).  
     3. Common pitfalls and debugging tips.

2. **Automated Testing & CI**  
   - Configure **GitHub Actions** to run:  
     - `pytest` on every PR, covering:  
       - Module contract validations (ensure `module.yaml` has required fields).  
       - Flow Runner integration tests.  
       - Basic FastAPI health endpoint test.  
   - Ensure coverage reports generate, and PRs cannot merge if tests fail.

3. **Internal Demo & Feedback Loop**  
   - Schedule a 30–60 minute **demo meeting** with:  
     - Product Manager  
     - Legal SMEs  
     - QA/Operations  
   - Walk through:  
     1. Module registry discovery (`GET /api/modules`).  
     2. Agent Builder UI: choose modules, fill config, run flow.  
     3. Show final output (drafted contract + compliance report).  
   - Collect improvement notes:  
     - “Template module should support clause toggles.”  
     - “Need richer compliance rules (negotiable vs. mandatory).”  
   - Triage feedback and file follow-up tickets.

4. **Deliverables**  
   - **Updated README.md** with full MVP 1 instructions.  
   - **DEVELOPER_GUIDE.md** in backend.  
   - **GitHub Actions** config for automated tests.  
   - **Demo PPT/Recording** outlining how to build and run a modular Agent.

---

## Repository Structure

```plaintext
juris/
├── backend                  # FastAPI server, module code, and flow logic
│   ├── app                  # Core application (main.py, routers, service layer)
│   ├── core                 # Flow runner, module loader, utility functions
│   ├── modules              # All modular Agent code & metadata
│   │   ├── retrieval        # retrieval module folder
│   │   │   ├── v1.0.0
│   │   │   │   ├── module.yaml
│   │   │   │   └── implementation.py
│   │   │   └── v1.1.0
│   │   ├── template         # template/drafting module folder
│   │   │   ├── v0.1.0
│   │   │   │   ├── module.yaml
│   │   │   │   ├── implementation.py
│   │   │   │   └── templates/  # Jinja2 or similar templates
│   │   │   └── v0.1.1
│   │   └── compliance       # compliance module folder
│   │       ├── v0.1.0
│   │       │   ├── module.yaml
│   │       │   └── implementation.py
│   │       └── v0.2.0
│   ├── config.py            # Settings loader (env, secrets)
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # For containerizing backend
├── frontend                 # React application (Agent Builder, Search UI)
│   ├── src
│   │   ├── components       # Reusable UI components
│   │   ├── pages            # Home, AgentBuilder.jsx, etc.
│   │   ├── services         # API client (fetch modules, run flow)
│   │   └── App.jsx
│   ├── public
│   └── package.json
├── tests                    # Pytest files for backend modules & flow runner
│   ├── test_retrieval.py
│   ├── test_template.py
│   ├── test_compliance.py
│   └── test_flow_runner.py
├── docker-compose.yml       # Orchestrates backend, frontend, and optional vector DB
├── .github                  # GitHub Actions workflows
│   └── ci.yml
└── README.md                # This file
```


---

## Prerequisites

- Docker & Docker Compose (for full-stack local dev)

- Git 2.x

- Node.js & npm (for React frontend)

- Python 3.10+

- (Optional) Access to an OpenAI API key when switching to a cloud LLM for Compliance or Drafting modules.

## Installation

### Clone the Repository
```bash
git clone https://github.com/Jackson00Han/juris.git
cd juris/backend
```

### Backend Setup (Optional if using Docker)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend Setup (Optional if using Docker)
```bash
cd ../frontend
npm install
```


## Configuration

- Copy any .env.example templates for backend or first_try:
```
cp backend/.env.example backend/.env
```
- Edit backend/.env to set environment variables (e.g. PORT, vector DB credentials, LLM API keys).

## Sprints

### Sprint 0: Audit & Requirements (1 week)
Review RAG infrastructure, define modules, gather use cases.

### Sprint 1: Framework Design (2 weeks)
Define module contract (YAML), build Module Loader & Orchestration Engine skeleton.

### Sprint 2: Core Module Implementation (3-4 weeks)
Build Retrieval, Template, Compliance modules, plus unit tests.  
Develop Agent Builder UI prototype in React.

### Sprint 3: Integration & Demo (2 weeks)
End-to-end flows, internal demos, gather feedback, refine UX.

### Sprint 4: Docs & CI (1–2 weeks)
Finalize README, Developer Guide, automated tests, and prepare for internal launch.

## Running with Docker Compose

When all components are ready, bring up the full stack (backend, frontend, plus optional vector DB):
```
docker-compose up --build
```
  - Backend -> http://localhost:8000
  - Frontend -> http://localhost:3000

(If a Docker service hasn't been built for Retrieval/Embedding DB yet, you can skip the vector DB service or comment it out.)

## Development Workflow

### Backend (FastAPI)
```
cd backend  
source .venv/bin/activate  
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000  
```
#### The API exposes:

- GET / → health check + brief welcome message  
- GET /api/modules → list available modules + versions  
- POST /api/flow/run → accepts agent_config.json, returns flow output JSON  

---

### Frontend (React)
```
cd frontend  
npm start  
```
Access UI at http://localhost:3000, including the “Agent Builder” page.



## Branching Strategy

- main: Protected. Holds only stable releases. Merges by PR.

- dev: Integration branch for active development.

- Feature branches:

  - Naming: feature/<short-description>

  - Base off dev, open PRs into dev for review.

  - Once dev is stable, merge into main.


## License

Copyright © 2025 Shanzhong Han

This project is provided for personal learning and research purposes only.  
Unauthorized copying, modification, distribution, or commercial use is strictly prohibited.  
For permission to use this project, please contact the author.
