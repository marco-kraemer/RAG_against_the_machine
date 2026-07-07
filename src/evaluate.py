from pathlib import Path
from typing import Dict, Iterable, List

from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

RAW_REPO_PREFIX = "data/raw/vllm-0.10.1/"
MINIMUM_OVERLAP_RATIO = 0.05


def normalize_source_path(path: str) -> str:
    """Normalize source paths so indexed and ground-truth
    paths compare equally."""
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith(RAW_REPO_PREFIX):
        normalized = normalized[len(RAW_REPO_PREFIX):]
    return normalized


def source_overlap_ratio(
    expected: MinimalSource,
    retrieved: MinimalSource,
) -> float:
    """Return overlap length divided by the expected source length."""
    if normalize_source_path(expected.file_path) != normalize_source_path(
        retrieved.file_path
    ):
        return 0.0

    expected_length = (
        expected.last_character_index - expected.first_character_index
    )
    if expected_length <= 0:
        return 0.0

    overlap_start = max(
        expected.first_character_index,
        retrieved.first_character_index,
    )
    overlap_end = min(
        expected.last_character_index,
        retrieved.last_character_index,
    )
    overlap_length = max(0, overlap_end - overlap_start)
    return overlap_length / expected_length


def source_found(
    expected: MinimalSource,
    retrieved_sources: Iterable[MinimalSource],
) -> bool:
    """Return whether a ground-truth source is found in retrieved sources."""
    return any(
        source_overlap_ratio(expected, retrieved) >= MINIMUM_OVERLAP_RATIO
        for retrieved in retrieved_sources
    )


def recall_for_question(
    expected_sources: List[MinimalSource],
    retrieved_sources: List[MinimalSource],
) -> float:
    """Compute recall for one question using subject source-overlap rules."""
    if not expected_sources:
        return 0.0

    found_count = sum(
        1
        for expected in expected_sources
        if source_found(expected, retrieved_sources)
    )
    return found_count / len(expected_sources)


def _load_dataset(dataset_path: Path) -> RagDataset:
    """Load and validate a ground-truth RAG dataset from JSON."""
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        return RagDataset.model_validate_json(dataset_file.read())


def _load_search_results(search_results_path: Path) -> StudentSearchResults:
    """Load and validate saved student search results from JSON."""
    with search_results_path.open("r", encoding="utf-8") as results_file:
        return StudentSearchResults.model_validate_json(results_file.read())


def _metric_names(k: int) -> List[int]:
    """Return the ordered, de-duplicated k values to report recall for."""
    requested_ks = [1, 3, 5, k]
    metric_ks: List[int] = []
    for value in requested_ks:
        if value > 0 and value not in metric_ks:
            metric_ks.append(value)
    return metric_ks


def _student_results_by_id(
    student_results: StudentSearchResults,
) -> Dict[str, MinimalSearchResults]:
    """Index student search results by their question id for lookup."""
    return {
        result.question_id: result
        for result in student_results.search_results
    }


def evaluate_search_results(
    student_search_results_path: str,
    dataset_path: str,
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate saved search results against answered
    ground-truth questions."""
    try:
        dataset = _load_dataset(Path(dataset_path))
        student_results = _load_search_results(
            Path(student_search_results_path)
        )
    except Exception as e:
        print(f"Error loading evaluation inputs: {e}")
        raise SystemExit(1) from e

    results_by_id = _student_results_by_id(student_results)
    answered_questions = [
        question
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion)
    ]
    questions_with_sources = [
        question for question in answered_questions if question.sources
    ]
    questions_with_student_sources = [
        question
        for question in questions_with_sources
        if results_by_id.get(question.question_id)
        and results_by_id[question.question_id].retrieved_sources
    ]

    metric_ks = _metric_names(k)
    totals = {metric_k: 0.0 for metric_k in metric_ks}

    for question in questions_with_sources:
        student_result = results_by_id.get(question.question_id)
        retrieved_sources = (
            student_result.retrieved_sources
            if student_result is not None
            else []
        )

        for metric_k in metric_ks:
            totals[metric_k] += recall_for_question(
                question.sources,
                retrieved_sources[:metric_k],
            )

    evaluated_count = len(questions_with_sources)
    metrics = {
        f"Recall@{metric_k}": (
            totals[metric_k] / evaluated_count if evaluated_count else 0.0
        )
        for metric_k in metric_ks
    }

    print("Student data is valid: True")
    print(f"Total number of questions: {len(answered_questions)}")
    print(
        "Total number of questions with sources: "
        f"{len(questions_with_sources)}"
    )
    print(
        "Total number of questions with student sources: "
        f"{len(questions_with_student_sources)}"
    )
    print("Evaluation Results")
    print("========================================")
    print(f"Questions evaluated: {evaluated_count}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.3f}")

    return metrics
