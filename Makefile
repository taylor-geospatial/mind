.PHONY: install check

install:
	uv sync

check:
	uv run pre-commit run --all-files
	uv run python -m unittest discover -s tests
