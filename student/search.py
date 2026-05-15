try:
    import json
    import sys
    import bm25s
    from typing import Dict, List
except ImportError:
    print("Run make install to install the required dependencies.")
    sys.exit(1)


def search(query: str, k: int):
    print(f"Searching for: {query} with top-k: {k}")
    try:
        retriever = bm25s.BM25.load(
            "./data/index/bm25_model",
            load_corpus=True,
        )
    except Exception as e:
        print(f"Error loading BM25 model: {e}")
        print("Run: uv run python -m student index")
        sys.exit(1)
    try:
        with open("./data/index/chunks_metadata.json", "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading chunks metadata: {e}")
        sys.exit(1)

    query_tokens = bm25s.tokenize(query, stopwords="en")
    docs, _ = retriever.retrieve(query_tokens, k=k)
    search_data: List[Dict] = []
    for match in docs[0]:
        id: int = match["id"]
        source = metadata[id]
        data: Dict = {}
        data["file_path"] = source["file_path"]
        data["first_character_index"] = source["first_character_index"]
        data["last_character_index"] = source["last_character_index"]
        search_data.append(data)

    # for data in search_data:
    #     print(data)

    return search_data
