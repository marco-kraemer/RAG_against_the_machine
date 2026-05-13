try:
    import json
    import re
    import sys

    import bm25s
    import fire

    from student.indexer import ChunkIndexer
except ImportError:
    print("Run make install to install the required dependencies.")
    print("Example: uv run python -m student")
    sys.exit(1)


class CLI:
    def index(self, max_chunk_size=2000):
        indexer = ChunkIndexer()
        indexer.index_repository(max_chunk_size=max_chunk_size)

    def search(self, query, k=10):
        print(f"Searching for: {query} with top-k: {k}")
        retriever = bm25s.BM25.load("./data/index/bm25_model", load_corpus=True)
        try:
            with open("./data/index/chunks_metadata.json", "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading chunks metadata: {e}")
            sys.exit(1)

        query_tokens = bm25s.tokenize(query, stopwords="en")
        docs, scores = retriever.retrieve(query_tokens, k=2)
        print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")

    def search_dataset(self, name, k=10):
        print(f"Searching dataset: {name} with top-k: {k}")

    def answer(self, question):
        print(f"Answering question: {question}")

    def answer_dataset(self, name, k=5):
        print(f"Answering dataset: {name} with top-k: {k}")

    def evaluate(self, model, dataset):
        print(f"Evaluating model: {model} on dataset: {dataset}")


if __name__ == "__main__":
    fire.Fire(CLI)
