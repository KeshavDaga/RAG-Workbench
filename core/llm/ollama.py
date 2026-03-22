import logging

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langfuse import get_client

from core.llm.base import Generator, Embedder
from core.observability import langchain_invoke_config

logger = logging.getLogger(__name__)

class OllamaGenerator(Generator):
    def __init__(self, model: str):
        self.model = model
        self.client = ChatOllama(model=self.model)
        self.parser = StrOutputParser()

    def generate(self, prompt: str) -> str:
        logger.info("OllamaGenerator model=%s invoke prompt_len=%d", self.model, len(prompt))
        response = self.client.invoke(prompt, config=langchain_invoke_config())
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
        lf = get_client()
        with lf.start_as_current_observation(
            name="OllamaEmbeddings.embed_documents",
            as_type="embedding",
            model=self.model_name,
            input={"batch_size": len(texts), "total_chars": total_chars},
        ) as obs:
            vectors = self.client.embed_documents(texts)
            dim = len(vectors[0]) if vectors else 0
            obs.update(output={"num_vectors": len(vectors), "dimension": dim})
        if vectors:
            logger.info(
                "OllamaEmbedder model=%s first_vector_dim=%d",
                self.model_name,
                len(vectors[0]),
            )
        return vectors

