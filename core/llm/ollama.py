from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from core.llm.base import Generator, Embedder
from langchain_core.output_parsers import StrOutputParser

class OllamaGenerator(Generator):
    def __init__(self, model: str):
        self.model = model
        self.client = ChatOllama(model=self.model)
        self.parser = StrOutputParser()

    def generate(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return self.parser.invoke(response)

class OllamaEmbedder(Embedder):
    def __init__(self, model: str):
        self.model_name = model
        self.client = OllamaEmbeddings(model=model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

