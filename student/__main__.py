try:
    import sys
    import fire
except ImportError:
    print("Run make install to install the required dependencies.")
    print("Example: uv run python -m student")
    sys.exit(1)


class CLI:
    def index(self):
        print("Hello, World!")

    def search(self, query, k=10):
        print(f"Searching for: {query} with top-k: {k}")

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
