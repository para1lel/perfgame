.PHONY: setup dev check test-eval test-frontend build migrate fmt clean

CONDA_MAIN ?= perfgame-main
CONDA_TIRX ?= perfgame-tirx
PY := conda run -n $(CONDA_MAIN) python

setup:
	conda env create -f environment-main.yml || conda env update -f environment-main.yml --prune
	conda env create -f environment-tirx.yml || conda env update -f environment-tirx.yml --prune
	npm ci --prefix frontend
	mkdir -p data/groundtruth data/artifacts data/profiles
	$(PY) -m manage fetch-cutlass
	$(PY) -m manage generate-groundtruth --all

dev:
	npm run dev --prefix frontend & \
	$(PY) -m uvicorn backend.main:app --reload --port 8000

check:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	npm run lint --prefix frontend
	npm run typecheck --prefix frontend
	$(PY) -m pytest tests -q

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .
	npm run format --prefix frontend

# 需要一张空闲 GPU，会占用数分钟
test-eval:
	$(PY) -m pytest tests/eval -q -m gpu
	$(PY) -m manage smoke-test --gpu 0

test-frontend:
	npm run test --prefix frontend

build:
	npm run build --prefix frontend

migrate:
	$(PY) -m manage migrate

clean:
	rm -rf frontend/dist data/artifacts data/profiles
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
