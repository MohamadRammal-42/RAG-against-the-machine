PYTHON      := python3
MAIN        := main.py
QUESTION    := "What flag must be explicitly passed when serving VLM2Vec-Full model in vLLM to run it in embedding mode?"

.DEFAULT_GOAL := run

install:
	uv sync

debug:
	uv run python3 -m pdb -m src answer $(QUESTION)

run:
	uv run python3 -m src answer $(QUESTION)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	uv run flake8 . --exclude=.venv,data
	uv run mypy . \
	--exclude '(^|/)\.venv(/|$$)' \
	--exclude '(^|/)data(/|$$)' \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	mypy . --strict --exclude '(^|/)\.venv(/|$$)'

index:
	uv run python3 -m src index

moul:
	uv run python3 -m src search_dataset --dataset_path "data/datasets/public/AnsweredQuestions/dataset_docs_public.json" --save_directory "data/output/search_results"
	uv run python3 -m src search_dataset --dataset_path "data/datasets/public/AnsweredQuestions/dataset_code_public.json" --save_directory "data/output/search_results"
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/dataset_docs_public.json  data/datasets/public/AnsweredQuestions/dataset_docs_public.json
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/dataset_code_public.json  data/datasets/public/AnsweredQuestions/dataset_code_public.json