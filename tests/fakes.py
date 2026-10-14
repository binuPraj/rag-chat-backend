"""Test doubles, so tests never download a model or need running servers."""

import math
import re
import zlib


class FakeEmbedder:
    dimension: int = 64

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for word in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)