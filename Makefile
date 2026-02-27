.PHONY: install lint test run

install:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .

lint:
	ruff check .

test:
	pytest -q

run:
	. .venv/bin/activate && aqf run
