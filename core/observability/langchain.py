"""LangChain invoke config: fresh Langfuse callback per call (safe under concurrency)."""

from langfuse.langchain import CallbackHandler


def langchain_invoke_config() -> dict:
    """Return LangChain RunnableConfig for ChatOllama.invoke with Langfuse tracing."""
    return {"callbacks": [CallbackHandler()]}
