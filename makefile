SHELL := zsh

.PHONY: lint test coverage commit docs

lint:
	uv run pre-commit run -a

test:
	uv run pytest

coverage:
	uv run coverage run
	uv run coverage report
	uv run coverage html

commit: test
	uv run cz c

docs:
	uv run make -C docs html
