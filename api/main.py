from fastapi import FastAPI

from api.routes import router

app = FastAPI(title="rag-workbench")

app.include_router(router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}