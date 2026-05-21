import pytest
from pydantic import ValidationError

from student.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_unanswered_question_generates_question_id() -> None:
    question = UnansweredQuestion(question="What is RAG?")

    assert question.question == "What is RAG?"
    assert isinstance(question.question_id, str)
    assert question.question_id


def test_answered_question_accepts_sources_and_answer() -> None:
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/index.md",
        first_character_index=10,
        last_character_index=50,
    )

    question = AnsweredQuestion(
        question_id="q1",
        question="Where is the docs index?",
        sources=[source],
        answer="The docs index is in docs/index.md.",
    )

    assert question.model_dump() == {
        "question_id": "q1",
        "question": "Where is the docs index?",
        "sources": [
            {
                "file_path": "data/raw/vllm-0.10.1/docs/index.md",
                "first_character_index": 10,
                "last_character_index": 50,
            }
        ],
        "answer": "The docs index is in docs/index.md.",
    }


def test_rag_dataset_accepts_answered_and_unanswered_questions() -> None:
    dataset = RagDataset(
        rag_questions=[
            UnansweredQuestion(question_id="q1", question="Question only?"),
            AnsweredQuestion(
                question_id="q2",
                question="Answered?",
                sources=[
                    MinimalSource(
                        file_path="file.py",
                        first_character_index=0,
                        last_character_index=10,
                    )
                ],
                answer="Yes.",
            ),
        ]
    )

    assert len(dataset.rag_questions) == 2
    assert dataset.rag_questions[0].question_id == "q1"
    assert isinstance(dataset.rag_questions[1], AnsweredQuestion)


def test_student_search_results_uses_subject_shape() -> None:
    result = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="How does retrieval work?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="student/search.py",
                        first_character_index=1,
                        last_character_index=25,
                    )
                ],
            )
        ],
        k=10,
    )

    assert result.model_dump() == {
        "search_results": [
            {
                "question_id": "q1",
                "question": "How does retrieval work?",
                "question_str": "How does retrieval work?",
                "retrieved_sources": [
                    {
                        "file_path": "student/search.py",
                        "first_character_index": 1,
                        "last_character_index": 25,
                    }
                ],
            }
        ],
        "k": 10,
    }


def test_student_search_results_and_answer_accepts_answers() -> None:
    results = StudentSearchResultsAndAnswer(
        search_results=[
            MinimalAnswer(
                question_id="q1",
                question="How does generation work?",
                retrieved_sources=[],
                answer="It uses retrieved context.",
            )
        ],
        k=5,
    )

    assert results.search_results[0].answer == "It uses retrieved context."


def test_minimal_source_rejects_invalid_types() -> None:
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="student/search.py",
            first_character_index="start",
            last_character_index=25,
        )
