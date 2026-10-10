.PHONY: install test quick run

install:
	pip install -r requirements-dev.txt

test:
	pytest -q

quick:
	python scripts/run_experiments.py --quick

run:
	python scripts/run_experiments.py
