from core.llm.base import Generator, Embedder
from core.llm.ollama import OllamaGenerator, OllamaEmbedder

GENERATORS = {
    "qwen": {
        "provider": "ollama",
        "model": "qwen3.5:35b",
    },
    "gemma3": {
        "provider": "ollama",
        "model": "gemma3:12b",
    },
}

EMBEDDERS = {
    "qwen3": {
        "provider": "ollama",
        "model": "qwen3-embedding:latest",
    },
}


def get_generator(key: str) -> Generator:
    cfg = GENERATORS[key]
    if cfg["provider"] == "ollama":
        return OllamaGenerator(cfg["model"])

    raise ValueError(f"Unsupported generator: {key}")


def get_embedder(key: str) -> Embedder:
    cfg = EMBEDDERS[key]
    if cfg["provider"] == "ollama":
        return OllamaEmbedder(cfg["model"])

    raise ValueError(f"Unsupported embedder: {key}")
