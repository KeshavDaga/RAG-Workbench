# RAG-Workbench

## Architecture overview

- **API layer** (FastAPI)
- **Core engine** (RAG orchestration)
- **Retriever** (query embedding + similarity search)
- **Vector store** (FAISS)
- **LLM providers** (Ollama)

## Running locally

### Requirements

- Python 3.12
- Ollama installed and running

### Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the server

```bash
uvicorn api.main:app --reload
```

### API documentation

Once the server starts, you can access the interactive API documentation at:

```
http://127.0.0.1:8000/docs
```
