import logging
import math

from langfuse import observe

from core.llm.base import Generator

logger = logging.getLogger(__name__)

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


@observe(
    name="hierarchical_summarize",
    as_type="chain",
    # Avoid attaching full chunk texts to Langfuse (large / sensitive).
    capture_input=False,
    capture_output=True,
)
def hierarchical_summarize(generator: Generator, chunk_texts: list[str]) -> str:
    """
    Map: every CHUNKS_PER_MAP chunks -> one summary.
    Reduce: while more than 10 summaries, merge in groups of
    merge_size = min(ceil(n / 10), 5).
    Final: one LLM call to produce the user-facing summary (if more than one block left).
    """
    if not chunk_texts:
        raise ValueError("No transcript chunks to summarize")

    n_chunks = len(chunk_texts)
    n_map_batches = math.ceil(n_chunks / CHUNKS_PER_MAP)
    logger.info(
        "hierarchical_summarize map phase: %d chunks -> ~%d map calls (batch_size=%d)",
        n_chunks,
        n_map_batches,
        CHUNKS_PER_MAP,
    )

    summaries: list[str] = []
    for i in range(0, len(chunk_texts), CHUNKS_PER_MAP):
        batch = chunk_texts[i : i + CHUNKS_PER_MAP]
        batch_idx = i // CHUNKS_PER_MAP + 1
        logger.info(
            "hierarchical_summarize map batch %d/%d (chunks %d-%d)",
            batch_idx,
            n_map_batches,
            i,
            min(i + len(batch), n_chunks) - 1,
        )
        summaries.append(generator.generate(_chunk_map_prompt(batch)))

    reduce_round = 0
    while len(summaries) > 10:
        reduce_round += 1
        n = len(summaries)
        merge_size = min(math.ceil(n / 10), 5)
        logger.info(
            "hierarchical_summarize reduce round %d: merging %d summaries merge_size=%d",
            reduce_round,
            n,
            merge_size,
        )
        new_summaries: list[str] = []
        for j in range(0, n, merge_size):
            batch = summaries[j : j + merge_size]
            new_summaries.append(generator.generate(_reduce_prompt(batch)))
        summaries = new_summaries
        logger.info(
            "hierarchical_summarize reduce round %d done -> %d summaries",
            reduce_round,
            len(summaries),
        )

    if len(summaries) == 1:
        logger.info("hierarchical_summarize final: single summary, skipping merge")
        return summaries[0]
    logger.info(
        "hierarchical_summarize final merge: %d section summaries -> one",
        len(summaries),
    )
    return generator.generate(_final_prompt(summaries))
