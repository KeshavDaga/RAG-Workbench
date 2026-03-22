class Chunker:
    @staticmethod
    def chunk_segments(
        segments: list[dict],
        max_chars: int = 2000,
        overlap_ratio: float = 0.1,
    ) -> list[dict]:
        """
        Merge transcript segments into chunks suitable for embedding.

        - max_chars: maximum characters per chunk
        - overlap_ratio: percentage of segments to overlap between chunks
        """
        chunks = []
        current_segments = []

        for segment in segments:
            # Tentative text size if we add this segment
            tentative_text = " ".join(
                s["text"] for s in current_segments + [segment]
            )

            if len(tentative_text) > max_chars and current_segments:
                # Finalize current chunk
                start = current_segments[0]["start"]
                last = current_segments[-1]
                end = last["start"] + last["duration"]

                chunks.append({
                    "text": " ".join(s["text"] for s in current_segments),
                    "start": start,
                    "end": end,
                })

                # Compute segment-based overlap
                overlap_segments = max(
                    1,
                    int(len(current_segments) * overlap_ratio)
                )

                current_segments = current_segments[-overlap_segments:]

            current_segments.append(segment)

        # Flush remaining segments
        if current_segments:
            start = current_segments[0]["start"]
            last = current_segments[-1]
            end = last["start"] + last["duration"]

            chunks.append({
                "text": " ".join(s["text"] for s in current_segments),
                "start": start,
                "end": end,
            })

        return chunks
