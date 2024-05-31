SHELL := /usr/bin/env zsh

.PHONY: lint test coverage commit docs

lint:
	pre-commit run -a

test:
	pytest

coverage:
	coverage run
	coverage report

commit: test
	cz c

docs:
	cd docs && make html
