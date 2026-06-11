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
	@echo "Evaluation workflow:"
	@echo "  make install      Install dependencies (uv sync)"
	@echo "  make index        Build the BM25 index (run before searching)"
	@echo "  make search-all   Generate docs + code search results"
	@echo "  make moulinette   Grade results (docs >=0.80, code >=0.50)"
	@echo "  make pipeline     index + search-all + moulinette, end to end"
	@echo ""
	@echo "Demo a single query:"
	@echo "  make search       One sample search query"
	@echo "  make answer       One sample answer query (loads the LLM)"
	@echo ""
	@echo "Quality:"
	@echo "  make lint         Run flake8 + mypy"
	@echo "  make clean        Remove generated index, outputs, caches"

install:
	uv sync

run: pipeline

debug:
	$(PYTHON) -m pdb -m student search "How to configure OpenAI server?" -k 5

clean:
	rm -rf .mypy_cache __pycache__ student/__pycache__
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

evaluate: evaluate-docs evaluate-code

moulinette-docs:
	$(MOULINETTE) evaluate_student_search_results $(DOCS_RESULTS) $(DOCS_ANSWERED) --k $(K) --max_context_length $(MAX_CONTEXT_LENGTH) --threshold 0.80

moulinette-code:
	$(MOULINETTE) evaluate_student_search_results $(CODE_RESULTS) $(CODE_ANSWERED) --k $(K) --max_context_length $(MAX_CONTEXT_LENGTH) --threshold 0.50

moulinette: moulinette-docs moulinette-code

pipeline: index search-all moulinette evaluate
