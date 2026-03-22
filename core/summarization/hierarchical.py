import math

from core.llm.base import Generator

CHUNKS_PER_MAP = 5


def _chunk_map_prompt(blocks: list[str]) -> str:
    joined = "\n\n---\n\n".join(
        f"[Part {i + 1}]\n{t}" for i, t in enumerate(blocks)
    )
    return f"""You are summarizing a section of a video transcript. Summarize the following parts in order. Be concise; keep key facts and topics.

{joined}

Section summary:"""


def _reduce_prompt(summaries: list[str]) -> str:
    joined = "\n\n".join(
        f"[Summary {i + 1}]\n{s}" for i, s in enumerate(summaries)
    )
    return f"""Merge these partial summaries into one coherent summary. Remove redundancy; preserve order of topics when it matters.

{joined}

Merged summary:"""


def _final_prompt(summaries: list[str]) -> str:
    joined = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(summaries))
    return f"""Write a single clear final summary of the full video for a reader who has not watched it. Use the section summaries below.

{joined}

Final video summary:"""


def hierarchical_summarize(generator: Generator, chunk_texts: list[str]) -> str:
    """
    Map: every CHUNKS_PER_MAP chunks -> one summary.
    Reduce: while more than 10 summaries, merge in groups of
    merge_size = min(ceil(n / 10), 5).
    Final: one LLM call to produce the user-facing summary (if more than one block left).
    """
    if not chunk_texts:
        raise ValueError("No transcript chunks to summarize")

    summaries: list[str] = []
    for i in range(0, len(chunk_texts), CHUNKS_PER_MAP):
        batch = chunk_texts[i : i + CHUNKS_PER_MAP]
        summaries.append(generator.generate(_chunk_map_prompt(batch)))

    while len(summaries) > 10:
        n = len(summaries)
        merge_size = min(math.ceil(n / 10), 5)
        new_summaries: list[str] = []
        for j in range(0, n, merge_size):
            batch = summaries[j : j + merge_size]
            new_summaries.append(generator.generate(_reduce_prompt(batch)))
        summaries = new_summaries

    if len(summaries) == 1:
        return summaries[0]
    return generator.generate(_final_prompt(summaries))
