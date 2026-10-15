import pytest

from app.services.chunking import (
    ChunkStrategy,
    FixedSizeChunker,
    SentenceChunker,
    get_chunker,
)


def test_fixed_empty_text_returns_no_chunks() -> None:
    assert FixedSizeChunker().chunk("   ") == []


def test_fixed_chunks_respect_size_and_overlap() -> None:
    text = "a" * 250
    chunks = FixedSizeChunker(chunk_size=100, overlap=20).chunk(text)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) == 3
    assert "".join(c[:80] for c in chunks[:-1]) + chunks[-1] == text[: 80 * 2] + chunks[-1]


def test_fixed_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=50, overlap=50)


def test_sentence_never_splits_mid_sentence() -> None:
    text = "First sentence here. Second one follows! Is this third? Yes it is."
    chunks = SentenceChunker(max_chars=45).chunk(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.endswith((".", "!", "?"))


def test_sentence_overlaps_one_sentence() -> None:
    text = "Alpha is first. Bravo is second. Charlie is third. Delta is fourth."
    chunks = SentenceChunker(max_chars=35).chunk(text)
    assert chunks[0].split(". ")[-1].rstrip(".") in chunks[1]


def test_factory_returns_requested_strategy() -> None:
    assert isinstance(get_chunker(ChunkStrategy.FIXED), FixedSizeChunker)
    assert isinstance(get_chunker(ChunkStrategy.SENTENCE), SentenceChunker)
