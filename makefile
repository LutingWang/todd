SHELL := /usr/bin/env zsh

.PHONY: lint test coverage commit docs

lint:
	pipenv run pre-commit run -a

test:
	pipenv run pytest

coverage:
	pipenv run coverage run
	pipenv run coverage report
	pipenv run coverage html

commit: test
	pipenv run cz c

docs:
	cd docs && PIPENV_PIPFILE=../Pipfile pipenv run make html
