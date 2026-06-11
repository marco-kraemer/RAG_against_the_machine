PYTHON = uv run python
STUDENT = $(PYTHON) -m student
MOULINETTE = ./moulinette/moulinette-ubuntu
SRC = student/__init__.py student/__main__.py student/answer.py \
	student/evaluate.py student/indexer.py student/models.py student/search.py

DOCS_UNANSWERED = data/datasets/public/UnansweredQuestions/dataset_docs_public.json
DOCS_ANSWERED = data/datasets/public/AnsweredQuestions/dataset_docs_public.json
DOCS_RESULTS = data/output/search_results/dataset_docs_public.json

CODE_UNANSWERED = data/datasets/public/UnansweredQuestions/dataset_code_public.json
CODE_ANSWERED = data/datasets/public/AnsweredQuestions/dataset_code_public.json
CODE_RESULTS = data/output/search_results/dataset_code_public.json

K = 10
MAX_CONTEXT_LENGTH = 2000

.PHONY: help install run debug clean lint lint-strict answer search index \
	search-docs search-code search-all search-dataset answer-dataset \
	evaluate-docs evaluate-code moulinette-docs moulinette-code moulinette \
	pipeline

help:
	@echo "make install          Install dependencies"
	@echo "make index            Index vLLM into data/processed"
	@echo "make search           Run one sample search query"
	@echo "make answer           Run one sample answer query"
	@echo "make search-docs      Generate docs search results"
	@echo "make search-code      Generate code search results"
	@echo "make search-all       Generate docs and code search results"
	@echo "make moulinette-docs  Check docs results with official moulinette"
	@echo "make moulinette-code  Check code results with official moulinette"
	@echo "make moulinette       Check both docs and code results"
	@echo "make pipeline         Run index, search-all, and moulinette"
	@echo "make clean            Remove generated outputs and caches"
	@echo "make lint             Run flake8 and mypy"

install:
	uv sync

run: search

debug:
	$(PYTHON) -m pdb -m student search "How to configure OpenAI server?" -k 5

clean:
	rm -rf .pytest_cache .mypy_cache __pycache__ student/__pycache__
	rm -rf data/processed data/output

lint:
	uv run flake8 $(SRC)
	uv run mypy $(SRC) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 $(SRC)
	uv run mypy $(SRC) --strict

answer:
	$(STUDENT) answer "How to configure OpenAI server?" --k $(K)

search:
	$(STUDENT) search "How to configure OpenAI server?" --k $(K)

index:
	$(STUDENT) index --max_chunk_size 2000

search-docs:
	$(STUDENT) search_dataset --dataset_path $(DOCS_UNANSWERED) --save_directory data/output/search_results --k $(K)

search-code:
	$(STUDENT) search_dataset --dataset_path $(CODE_UNANSWERED) --save_directory data/output/search_results --k $(K)

search-all: search-docs search-code

answer-dataset:
	$(STUDENT) answer_dataset --student_search_results_path $(DOCS_RESULTS)

evaluate-docs:
	$(STUDENT) evaluate --student_search_results_path $(DOCS_RESULTS) --dataset_path $(DOCS_ANSWERED) --k $(K)

evaluate-code:
	$(STUDENT) evaluate --student_search_results_path $(CODE_RESULTS) --dataset_path $(CODE_ANSWERED) --k $(K)

moulinette-docs:
	$(MOULINETTE) evaluate_student_search_results $(DOCS_RESULTS) $(DOCS_ANSWERED) --k $(K) --max_context_length $(MAX_CONTEXT_LENGTH) --threshold 0.80

moulinette-code:
	$(MOULINETTE) evaluate_student_search_results $(CODE_RESULTS) $(CODE_ANSWERED) --k $(K) --max_context_length $(MAX_CONTEXT_LENGTH) --threshold 0.50

moulinette: moulinette-docs moulinette-code

pipeline: index search-all moulinette
