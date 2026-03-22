import logging

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from core.llm.base import Generator, Embedder
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class OllamaGenerator(Generator):
    def __init__(self, model: str):
        self.model = model
        self.client = ChatOllama(model=self.model)
        self.parser = StrOutputParser()

    def generate(self, prompt: str) -> str:
        logger.info("OllamaGenerator model=%s invoke prompt_len=%d", self.model, len(prompt))
        response = self.client.invoke(prompt)
        out = self.parser.invoke(response)
        logger.info("OllamaGenerator model=%s response_len=%d", self.model, len(out))
        return out

class OllamaEmbedder(Embedder):
    def __init__(self, model: str):
        self.model_name = model
        self.client = OllamaEmbeddings(model=model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        total_chars = sum(len(t) for t in texts)
        logger.info(
            "OllamaEmbedder model=%s embed_documents count=%d total_chars=%d",
            self.model_name,
            len(texts),
            total_chars,
        )
        vectors = self.client.embed_documents(texts)
        if vectors:
            logger.info(
                "OllamaEmbedder model=%s first_vector_dim=%d",
                self.model_name,
                len(vectors[0]),
            )
        return vectors

