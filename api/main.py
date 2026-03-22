
from dotenv import load_dotenv

load_dotenv()

import logging

from fastapi import FastAPI

from api.routes import router

# Ensure app modules (api, core, ingestion) log at INFO when running under uvicorn.
_root = logging.getLogger()
if not _root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
else:
    for _pkg in ("api", "core", "ingestion"):
        logging.getLogger(_pkg).setLevel(logging.INFO)

app = FastAPI(title="rag-workbench")

app.include_router(router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}