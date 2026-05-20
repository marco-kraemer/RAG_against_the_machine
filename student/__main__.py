try:
    import json
    import sys

    import fire

    from student.answer import answer_dataset, answer_query
    from student.indexer import ChunkIndexer
    from student.search import search, search_dataset
except ImportError:
    print("Run make install to install the required dependencies.")
    print("Example: uv run python -m student")
    sys.exit(1)


class CLI:
    def index(self, max_chunk_size=2000):
        indexer = ChunkIndexer()
        indexer.index_repository(max_chunk_size=max_chunk_size)

    def search(self, query, k=10):
        result = search(query, k)
        print(json.dumps(result.model_dump(), indent=2))

    def search_dataset(
        self,
        dataset_path,
        k=10,
        save_directory="data/output/search_results",
    ):
        search_dataset(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k,
        )

    def answer(self, query, k=10):
        try:
            result = answer_query(query, k=k)
        except RuntimeError as e:
            print(f"Error generating answer: {e}")
            sys.exit(1)
        print(json.dumps(result.model_dump(), indent=2))

    def answer_dataset(
        self,
        student_search_results_path,
        save_directory="data/output/search_results_and_answer",
    ):
        answer_dataset(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory,
        )

    def evaluate(self, model, dataset):
        print(f"Evaluating model: {model} on dataset: {dataset}")


if __name__ == "__main__":
    fire.Fire(CLI)
