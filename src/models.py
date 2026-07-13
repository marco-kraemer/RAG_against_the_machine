import uuid
from typing import List, Sequence

from pydantic import BaseModel, Field, computed_field


def validated_k(k: object) -> int:
    """Return k as an int, exiting gracefully when it is not one."""
    if isinstance(k, bool) or not isinstance(k, int):
        print(f"Error: k must be an integer, got {k!r}")
        raise SystemExit(1)
    return k


def validated_chunk_size(max_chunk_size: object) -> int:
    """Return max_chunk_size as a positive int, else exit gracefully."""
    if isinstance(max_chunk_size, bool) or not isinstance(max_chunk_size, int):
        print(f"Error: max_chunk_size must be an integer, "
              f"got {max_chunk_size!r}")
        raise SystemExit(1)
    if max_chunk_size <= 0:
        print(f"Error: max_chunk_size must be positive, got {max_chunk_size}")
        raise SystemExit(1)
    return max_chunk_size


class MinimalSource(BaseModel):
    """Minimal source location used by search and answer results."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question without a generated or reference answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with reference sources and an answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset containing answered or unanswered RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search result for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def question_str(self: "MinimalSearchResults") -> str:
        """Moulinette-compatible question field."""
        return self.question


class MinimalAnswer(MinimalSearchResults):
    """Search result enriched with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Top-k search results for a dataset."""

    search_results: Sequence[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Top-k search results enriched with generated answers."""

    search_results: Sequence[MinimalAnswer]
