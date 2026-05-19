import json

from student.models import MinimalSearchResults
from student.search import search


def _sample_chunks() -> list[dict[str, object]]:
    return [
        {
            "file_path": "docs/index.md",
            "first_character_index": 4,
            "last_character_index": 42,
            "content": "Internal context that should not be public.",
        },
        {
            "file_path": "vllm/config.py",
            "first_character_index": 100,
            "last_character_index": 180,
            "content": "More internal context.",
        },
    ]


def test_search_returns_subject_model_without_content(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("student.search._retrieve_chunks", lambda query, k: _sample_chunks())

    result = search("How does vLLM configure models?", k=2)

    assert isinstance(result, MinimalSearchResults)
    assert result.model_dump() == {
        "question_id": "cli_query",
        "question": "How does vLLM configure models?",
        "retrieved_sources": [
            {
                "file_path": "docs/index.md",
                "first_character_index": 4,
                "last_character_index": 42,
            },
            {
                "file_path": "vllm/config.py",
                "first_character_index": 100,
                "last_character_index": 180,
            },
        ],
    }
    assert capsys.readouterr().out == ""


def test_cli_search_prints_pretty_json(monkeypatch, capsys) -> None:
    from student.__main__ import CLI
    from student.models import MinimalSource

    def fake_search(query: str, k: int = 10) -> MinimalSearchResults:
        return MinimalSearchResults(
            question_id="cli_query",
            question=query,
            retrieved_sources=[
                MinimalSource(
                    file_path="docs/index.md",
                    first_character_index=4,
                    last_character_index=42,
                )
            ],
        )

    monkeypatch.setattr("student.__main__.search", fake_search)

    CLI().search("Where is the docs index?", k=1)

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "question_id": "cli_query",
        "question": "Where is the docs index?",
        "retrieved_sources": [
            {
                "file_path": "docs/index.md",
                "first_character_index": 4,
                "last_character_index": 42,
            }
        ],
    }
