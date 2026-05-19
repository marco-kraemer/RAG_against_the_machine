try:
    import json
    import sys
    import bm25s
    from typing import Any, Dict, List

    from student.models import MinimalSearchResults, MinimalSource
except ImportError:
    print("Run make install to install the required dependencies.")
    sys.exit(1)


_retriever = None
_metadata = None


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


def warmup_search() -> None:
    """Preload search resources before processing many queries."""
    get_retriever()
    get_metadata()


def _retrieve_chunks(query: str, k: int) -> List[Dict[str, Any]]:
    retriever = get_retriever()
    metadata = get_metadata()

    query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
    docs, _ = retriever.retrieve(query_tokens, k=k, show_progress=False)
    search_data: List[Dict[str, Any]] = []
    for match in docs[0]:
        id: int = match["id"]
        source = metadata[id]
        data: Dict[str, Any] = {}
        data["file_path"] = source["file_path"]
        data["first_character_index"] = source["first_character_index"]
        data["last_character_index"] = source["last_character_index"]
        data["content"] = source["content"]
        search_data.append(data)

    # for data in search_data:
    #     print(data)

    return search_data


def search(query: str, k: int = 10) -> MinimalSearchResults:
    chunks = _retrieve_chunks(query, k)
    return MinimalSearchResults(
        question_id="cli_query",
        question=query,
        retrieved_sources=[
            MinimalSource(
                file_path=chunk["file_path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"],
            )
            for chunk in chunks
        ],
    )
