import re

class Normalizer:
    @staticmethod
    def normalize_segments(segments: list[dict]) -> list[dict]:
        """
        Clean raw YouTube transcript segments.

        Guarantees:
        - text is stripped and non-empty
        - obvious non-speech markers are removed
        - ultra-short junk segments are dropped
        - timestamps are preserved
        """
        normalized = []

        for segment in segments:
            text = segment.get("text", "").strip()

            # Skip empty text
            if not text:
                continue

            # Skip non-speech markers like [Music], (Applause)
            if Normalizer._is_noise_marker(text):
                continue

            # Skip extremely short segments (likely filler)
            if len(text) < 3:
                continue

            normalized.append({
                "text": text,
                "start": segment["start"],
                "duration": segment["duration"],
            })

        return normalized

    @staticmethod
    def _is_noise_marker(text: str) -> bool:
        """
        Detect obvious non-speech transcript markers.
        Examples: [Music], (Applause), [Laughter]
        """
        return bool(
            re.fullmatch(r"\[.*\]", text) or
            re.fullmatch(r"\(.*\)", text)
        )
