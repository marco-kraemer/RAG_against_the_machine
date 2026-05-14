try:
    import json
    import sys

    import bm25s
except ImportError:
    print("Run make install to install the required dependencies.")
    sys.exit(1)


class Search:
    def search(self, query: str, k: int):
        print(f"Searching for: {query} with top-k: {k}")
        try:
            retriever = bm25s.BM25.load("./data/index/bm25_model", load_corpus=True)
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
        docs, scores = retriever.retrieve(query_tokens, k=k)
        for match in docs[0]:
            id: int = match["id"]
            source = metadata[id]
            print(f"\n--- Match ID: {id} ---")
            print(f"Content: {source['content']}")
            print(f"File: {source['file_path']}")
            print(
                f"Range: {source['first_character_index']} to {source['last_character_index']}"
            )
