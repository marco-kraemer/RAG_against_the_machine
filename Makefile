.PHONY: install run debug clean lint lint-strict answer search index search-dataset answer-dataset evaluate

install:
	uv sync

run:
	uv run python -m student search "How to configure OpenAI server?" -k 5

debug:
	uv run python -m pdb -m student search "How to configure OpenAI server?" -k 5

clean:
	rm -rf .pytest_cache .mypy_cache __pycache__ src/student/__pycache__
	rm -rf data/processed data/output

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

answer:
	uv run python -m student answer "How to configure OpenAI server?" --k 10

search:
	uv run python -m student search "How to configure OpenAI server?" --k 10

index: 
	uv run python -m student index

search-dataset:
	uv run python -m student search_dataset --dataset_path data/datasets/public/UnansweredQuestions/dataset_docs_public.json

answer-dataset:
	uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json

evaluate:
	uv run python -m student evaluate --student_search_results_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets/public/AnsweredQuestions/dataset_docs_public.json --k 10

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
