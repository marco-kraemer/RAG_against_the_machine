import uuid
from typing import List

from pydantic import BaseModel, Field, computed_field


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

    @computed_field
    @property
    def question_str(self) -> str:
        """Moulinette-compatible question field."""
        return self.question


class MinimalAnswer(MinimalSearchResults):
    """Search result enriched with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Top-k search results for a dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Top-k search results enriched with generated answers."""

    search_results: List[MinimalAnswer]
