.PHONY: install test build-index eval fetch-data build-metadata

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

fetch-data:
	python data/scripts/fetch_data.py

build-metadata:
	python data/scripts/merge_sources.py
	python data/scripts/generate_descriptions.py

build-index:
	python data/scripts/build_index.py

eval:
	python tests/eval/run_eval.py
