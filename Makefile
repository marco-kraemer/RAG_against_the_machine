PYTHON = uv run python
STUDENT = $(PYTHON) -m src
MOULINETTE = ./moulinette/moulinette-ubuntu

SEARCH_RESULTS_DIR = data/output/search_results/UnansweredQuestions
ANSWER_RESULTS_DIR = data/output/search_results_and_answer/UnansweredQuestions

DOCS_UNANSWERED = data/datasets/UnansweredQuestions/dataset_docs_public.json
DOCS_ANSWERED = data/datasets/AnsweredQuestions/dataset_docs_public.json
DOCS_RESULTS = $(SEARCH_RESULTS_DIR)/dataset_docs_public.json

CODE_UNANSWERED = data/datasets/UnansweredQuestions/dataset_code_public.json
CODE_ANSWERED = data/datasets/AnsweredQuestions/dataset_code_public.json
CODE_RESULTS = $(SEARCH_RESULTS_DIR)/dataset_code_public.json

K = 10
MAX_CONTEXT_LENGTH = 2000

.PHONY: help install run debug clean fclean lint lint-strict answer search \
	index search-docs search-code search-all search-dataset answer-dataset \
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
	@echo "  make fclean       clean + remove the .venv (full reset)"

install:
	uv sync

run: pipeline

debug:
	$(PYTHON) -m pdb -m src search "How to configure OpenAI server?" -k 5

clean:
	rm -rf .mypy_cache __pycache__ src/__pycache__
	rm -rf data/processed data/output

fclean: clean
	rm -rf .venv

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

answer:
	$(STUDENT) answer "How to configure OpenAI server?" --k $(K)

search:
	$(STUDENT) search "How to configure OpenAI server?" --k $(K)

index:
	$(STUDENT) index --max_chunk_size 2000

search-docs:
	$(STUDENT) search_dataset --dataset_path $(DOCS_UNANSWERED) --save_directory $(SEARCH_RESULTS_DIR) --k $(K)

search-code:
	$(STUDENT) search_dataset --dataset_path $(CODE_UNANSWERED) --save_directory $(SEARCH_RESULTS_DIR) --k $(K)

search-all: search-docs search-code

answer-dataset:
	$(STUDENT) answer_dataset --student_search_results_path $(DOCS_RESULTS) --save_directory $(ANSWER_RESULTS_DIR)

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

pipeline: index search-all moulinette
