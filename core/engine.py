import logging

from core.llm.base import Generator
from core.retriever.base import Retriever

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, generator: Generator, retriever: Retriever):
        self.generator = generator
        self.retriever = retriever

    def query(self, question: str) -> str:
        contexts = self.retriever.retrieve(question)
        logger.info(
            "RAG retrieve returned %d context(s) for question_len=%d",
            len(contexts),
            len(question),
        )
        prompt = self._build_prompt(contexts, question)
        logger.debug("RAG prompt total_len=%d", len(prompt))
        return self.generator.generate(prompt)


    def _build_prompt(self, contexts, question: str) -> str:
        context_text = "\n\n".join(
            f"[{i+1}] {ctx['text']}"
            for i, ctx in enumerate(contexts)
        )

        return f"""You are a precise assistant. Answer the question using ONLY the provided context.
        If the answer is not present, say "I don't know".

        Context:
        {context_text}

        Question:
        {question}

        Answer:
        """
