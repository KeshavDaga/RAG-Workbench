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

### Environment variables

Create a `.env` file in the project root and **do not commit** real secrets. Add your keys from [Langfuse Cloud](https://cloud.langfuse.com/) (or your self-hosted URL):

```bash
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

Use the US host if your project is in the US region: `https://us.cloud.langfuse.com`.

### Start the server

```bash
uvicorn api.main:app --reload
```

### API documentation

Once the server starts, you can access the interactive API documentation at:

```
http://127.0.0.1:8000/docs
```
