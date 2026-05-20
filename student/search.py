try:
    import json
    import sys
    from pathlib import Path
    from typing import Any, Dict, List

    import bm25s
    from tqdm import tqdm

    from student.models import (
        MinimalSearchResults,
        MinimalSource,
        RagDataset,
        StudentSearchResults,
    )
except ImportError:
    print("Run make install to install the required dependencies.")
    sys.exit(1)


_retriever = None
_metadata = None


def chunks_to_sources(chunks: List[Dict[str, Any]]) -> List[MinimalSource]:
    """Convert internal retrieved chunks to public source references."""
    return [
        MinimalSource(
            file_path=chunk["file_path"],
            first_character_index=chunk["first_character_index"],
            last_character_index=chunk["last_character_index"],
        )
        for chunk in chunks
    ]


def get_retriever():
    """Load and cache the BM25 retriever for warm in-process searches."""
    global _retriever
    if _retriever is None:
        try:
            _retriever = bm25s.BM25.load(
                "./data/index/bm25_model",
                load_corpus=True,
            )
        except Exception as e:
            print(f"Error loading BM25 model: {e}")
            print("Run: uv run python -m student index")
            sys.exit(1)
    return _retriever


def get_metadata() -> List[Dict[str, Any]]:
    """Load and cache chunk metadata for warm in-process searches."""
    global _metadata
    if _metadata is None:
        try:
            with open("./data/index/chunks_metadata.json", "r") as f:
                _metadata = json.load(f)
        except Exception as e:
            print(f"Error loading chunks metadata: {e}")
            sys.exit(1)
    return _metadata


def _retrieve_chunks(query: str, k: int) -> List[Dict[str, Any]]:
    """Internal function to retrieve top-k chunks for a query.
    BM25 returns document IDs and scores,
    we use the IDs to look up chunk metadata and content."""
    retriever = get_retriever()
    metadata = get_metadata()

    query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
    docs, _ = retriever.retrieve(query_tokens, k=k, show_progress=False)
    chunks: List[Dict[str, Any]] = []
    for match in docs[0]:
        id: int = match["id"]
        source = metadata[id]
        data: Dict[str, Any] = {}
        data["file_path"] = source["file_path"]
        data["first_character_index"] = source["first_character_index"]
        data["last_character_index"] = source["last_character_index"]
        data["content"] = source["content"]
        chunks.append(data)
    return chunks


def search(query: str, k: int = 10) -> MinimalSearchResults:
    chunks = _retrieve_chunks(query, k)
    return MinimalSearchResults(
        question_id="cli_query",
        question=query,
        retrieved_sources=chunks_to_sources(chunks),
    )


def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 10,
) -> Path:
    """Search every question in a dataset and save subject-shaped JSON."""
    input_path = Path(dataset_path)
    output_dir = Path(save_directory)
    output_path = output_dir / input_path.name

    try:
        with input_path.open("r", encoding="utf-8") as dataset_file:
            dataset = RagDataset.model_validate_json(dataset_file.read())
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        sys.exit(1)

    print(f"Loaded {len(dataset.rag_questions)} questions from {dataset_path}")

    search_results: List[MinimalSearchResults] = []
    for question in tqdm(dataset.rag_questions, desc="Searching questions"):
        chunks = _retrieve_chunks(question.question, k)
        search_results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=chunks_to_sources(chunks),
            )
        )

    student_results = StudentSearchResults(search_results=search_results, k=k)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            output_file.write(student_results.model_dump_json(indent=2))
            output_file.write("\n")
    except Exception as e:
        print(f"Error saving search results to {output_path}: {e}")
        sys.exit(1)

    print(f"Saved student_search_results to {output_path}")
    return output_path
