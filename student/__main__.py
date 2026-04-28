import fire


def index():
    print("Hello, World!")


def search(query, k=10):
    print(f"Searching for: {query} with top-k: {k}")


def search_dataset(name, k=10):
    print(f"Searching dataset: {name} with top-k: {k}")


def answer(question):
    print(f"Answering question: {question}")


def answer_dataset(name, k=5):
    print(f"Answering dataset: {name} with top-k: {k}")


def evaluate(model, dataset):
    print(f"Evaluating model: {model} on dataset: {dataset}")


if __name__ == "__main__":
    fire.Fire()
