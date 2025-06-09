.PHONY: install install-dev test lint format clean build

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	flake8 agent_eval tests
	mypy agent_eval tests
	black --check agent_eval tests
	isort --check-only agent_eval tests

format:
	black agent_eval tests
	isort agent_eval tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python setup.py sdist bdist_wheel 