import json

from student.models import MinimalSource


def test_source_found_matches_normalized_paths_with_minimum_overlap() -> None:
    from student.evaluate import source_found

    expected = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved = [
        MinimalSource(
            file_path="docs/features/lora.md",
            first_character_index=195,
            last_character_index=220,
        )
    ]

    assert source_found(expected, retrieved)


def test_evaluate_search_results_prints_recall_metrics(tmp_path, capsys) -> None:
    from student.evaluate import evaluate_search_results

    dataset_path = tmp_path / "answered.json"
    search_results_path = tmp_path / "search.json"

    dataset_path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {
                        "question_id": "q1",
                        "question": "Where is LoRA loading documented?",
                        "answer": "In the LoRA docs.",
                        "sources": [
                            {
                                "file_path": (
                                    "data/raw/vllm-0.10.1/docs/features/lora.md"
                                ),
                                "first_character_index": 100,
                                "last_character_index": 200,
                            }
                        ],
                    },
                    {
                        "question_id": "q2",
                        "question": "Where is quantization documented?",
                        "answer": "In the quantization docs.",
                        "sources": [
                            {
                                "file_path": (
                                    "data/raw/vllm-0.10.1/docs/features/"
                                    "quantization/int8.md"
                                ),
                                "first_character_index": 400,
                                "last_character_index": 500,
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    search_results_path.write_text(
        json.dumps(
            {
                "search_results": [
                    {
                        "question_id": "q1",
                        "question": "Where is LoRA loading documented?",
                        "retrieved_sources": [
                            {
                                "file_path": "docs/features/lora.md",
                                "first_character_index": 195,
                                "last_character_index": 220,
                            }
                        ],
                    },
                    {
                        "question_id": "q2",
                        "question": "Where is quantization documented?",
                        "retrieved_sources": [
                            {
                                "file_path": "docs/features/lora.md",
                                "first_character_index": 100,
                                "last_character_index": 200,
                            }
                        ],
                    },
                ],
                "k": 10,
            }
        ),
        encoding="utf-8",
    )

    metrics = evaluate_search_results(
        student_search_results_path=str(search_results_path),
        dataset_path=str(dataset_path),
        k=10,
    )

    assert metrics["Recall@1"] == 0.5
    output = capsys.readouterr().out
    assert "Student data is valid: True" in output
    assert "Questions evaluated: 2" in output
    assert "Recall@1: 0.500" in output
    assert "Recall@10: 0.500" in output
