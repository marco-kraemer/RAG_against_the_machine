.PHONY: install run debug clean lint lint-strict

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

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
