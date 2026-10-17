VENV   := .venv/bin
PYTHON := $(VENV)/python3
PIP    := $(VENV)/pip
# Training device: CUDA, then Apple Silicon (MPS), then CPU. Override with: make train DEVICE=cpu
DEVICE ?= $(shell $(PYTHON) -c "import torch; print(0 if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')" 2>/dev/null || echo cpu)

.PHONY: help venv install install-backend install-frontend install-dev backend frontend dev \
        dataset preprocess train test lint clean download-model

help:
	@echo "Usage:"
	@echo "  make install        - Create .venv and install backend + frontend dependencies"
	@echo "  make dev            - Start backend (:8000) + frontend (:9999) together"
	@echo "  make backend        - Start FastAPI server only (port 8000)"
	@echo "  make frontend       - Start Next.js dev server only (port 9999)"
	@echo "  make install-dev    - Install backend deps plus test and lint tooling"
	@echo ""
	@echo "  make dataset        - Install training deps and download HIT-UAV (needs Kaggle credentials)"
	@echo "  make preprocess     - Median + CLAHE + Unsharp -> dataset/hit-uav-processed"
	@echo "  make train          - Train YOLOv8n for 50 epochs on the processed dataset"
	@echo ""
	@echo "  make test           - Run API tests"
	@echo "  make lint           - Lint Python code (ruff) and type-check the frontend"
	@echo "  make clean          - Remove caches and build output"
	@echo "  make download-model - Show how to download best.pt from Colab"

venv:
	@test -d .venv || python3 -m venv .venv

install: venv install-backend install-frontend

install-backend: venv
	@echo "→ Installing backend dependencies..."
	$(PIP) install -q -r backend/requirements.txt

install-frontend:
	@echo "→ Installing frontend dependencies..."
	cd frontend && npm install

install-dev: install-backend
	$(PIP) install -q -r requirements-dev.txt

backend:
	@if [ ! -f backend/models/best.pt ]; then \
		echo "ERROR: backend/models/best.pt not found."; \
		echo "Run: make download-model"; \
		exit 1; \
	fi
	$(VENV)/uvicorn backend.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

dev:
	@if [ ! -f backend/models/best.pt ]; then \
		echo "ERROR: backend/models/best.pt not found."; \
		echo "Run: make download-model"; \
		exit 1; \
	fi
	@echo "→ Backend :8000  Frontend :9999"
	@trap 'kill 0' SIGINT; \
		$(VENV)/uvicorn backend.main:app --port 8000 & \
		cd frontend && npm run dev & \
		wait

dataset:
	./setup.sh

preprocess:
	$(PYTHON) scripts/preprocess.py

train:
	@test -d dataset/hit-uav-processed || { echo "Run: make preprocess"; exit 1; }
	$(VENV)/yolo detect train data=hituav_data.yaml model=yolov8n.pt epochs=50 imgsz=640 device=$(DEVICE) \
		project=runs/detect name=train exist_ok=True
	@echo "→ Copy runs/detect/train/weights/best.pt to backend/models/ to serve the new model"

test:
	$(PYTHON) -m pytest -q

lint:
	$(VENV)/ruff check backend scripts tests
	cd frontend && npx tsc --noEmit

clean:
	rm -rf .pytest_cache .ruff_cache frontend/.next
	find . -name __pycache__ -type d -prune -not -path "./.venv/*" -exec rm -rf {} +

download-model:
	@echo "Run this in Colab:"
	@echo ""
	@echo "  from google.colab import files"
	@echo "  files.download('/content/runs/detect/train/weights/best.pt')"
	@echo ""
	@echo "Then move it here:"
	@echo "  mv ~/Downloads/best.pt backend/models/"
