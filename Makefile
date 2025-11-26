.PHONY: help setup install update-reqs lab clean

help:
	@echo "Kaggle Competitions Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup        - Tạo conda environment từ environment.yml"
	@echo "  make install      - Cài đặt dependencies trong conda env"
	@echo "  make update-reqs  - Export requirements.txt từ conda env"
	@echo "  make lab          - Chạy Jupyter Lab"
	@echo "  make clean        - Dọn dẹp cache và temp files"

setup:
	@echo "Creating conda environment: kaggle-competitions"
	conda env create -f environment.yml
	@echo "✅ Environment created!"
	@echo "Activate with: conda activate kaggle-competitions"

install:
	@echo "Installing dependencies..."
	conda env update -f environment.yml --prune
	@echo "✅ Dependencies installed!"

update-reqs:
	@echo "Exporting requirements.txt from conda env..."
	@echo "# Auto-generated from conda env" > requirements-full.txt
	conda list --export >> requirements-full.txt
	@echo "✅ Exported to requirements-full.txt"
	@echo "⚠️  Note: Manually update requirements.txt with only necessary packages for Colab/Kaggle"

lab:
	@echo "Starting Jupyter Lab..."
	jupyter lab

clean:
	@echo "Cleaning cache and temp files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.tmp" -delete
	@echo "✅ Cleaned!"
