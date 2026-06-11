try:
    import json
    import sys

    import fire

    from student.answer import answer_dataset, answer_query
    from student.evaluate import evaluate_search_results
    from student.indexer import index_repository
    from student.search import search, search_dataset
except ImportError:
    print("Run make install to install the required dependencies.")
    print("Example: uv run python -m student")
    sys.exit(1)


class CLI:
    """Fire entry points for the RAG pipeline stages."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index the vLLM repository into data/processed."""
        index_repository(max_chunk_size=max_chunk_size)

    def search(self, query: str, k: int = 10) -> None:
        """Print the top-k search results for a single query."""
        result = search(query, k)
        print(json.dumps(result.model_dump(), indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search every question in a dataset and save the results."""
        search_dataset(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k,
        )

    def answer(self, query: str, k: int = 10) -> None:
        """Generate a grounded answer for a single query."""
        try:
            result = answer_query(query, k=k)
        except RuntimeError as e:
            print(f"Error generating answer: {e}")
            sys.exit(1)
        print(json.dumps(result.model_dump(), indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for previously saved search results."""
        answer_dataset(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory,
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Evaluate saved search results against ground-truth sources."""
        evaluate_search_results(
            student_search_results_path=student_search_results_path,
            dataset_path=dataset_path,
            k=k,
        )


if __name__ == "__main__":
    fire.Fire(CLI)
