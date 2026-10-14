import re
from enum import Enum
from typing import Protocol

class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"

class Chunker(Protocol):
    def chunk(self, text:str) -> list[str]:
        ...

class FixedSizeChunker:
    def __init__(self, chunk_size= 800, overlap:int =100):
        if chunk_size <=0:
            raise ValueError("chunk_size must be > 0")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >=0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text:str) -> list[str]:
        text=text.strip()
        if not text:
            return[]
        step = self.chunk_size-self.overlap 
        chunks: list[str] = []

        for start in range(0, len(text), step):
            piece= text[start: start+self.chunk_size].strip()
            if piece:
                chunks.append(piece)

            if start + self.chunk_size >= len(text):
                    break

        return chunks


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

class SentenceChunker:
    def __init__(self, max_chars: int = 800) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        self.max_chars=max_chars

    def chunk(self, text:str) -> list[str]:
        sentences = []
        pieces = _SENTENCE_SPLIT.split(text.strip())

        for s in pieces:
            if s.strip():
                sentences.append(s.strip())

        chunks: list[str] = []
        current: list[str] = []
        length = 0

        for sentence in sentences:
            added = len(sentence) + (1 if current else 0)
            if current and length + added > self.max_chars:
                chunks.append(" ".join(current))
                current = [current[-1]]
                length=len(current[0])
                added= len(sentence) +1

            current.append(sentence)
            length += added

        if current and (not chunks or " ".join(current) != chunks[-1]):
            chunks.append(" ".join(current))

        return chunks

def get_chunker(strategy: ChunkStrategy) -> Chunker:
    if strategy is ChunkStrategy.FIXED:
        return FixedSizeChunker()
    return SentenceChunker()
        


