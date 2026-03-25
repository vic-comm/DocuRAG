# DocuMind — Developer shortcuts
# Usage: make <target>

.PHONY: help install ui api demo test clean

help:
	@echo ""
	@echo "  DocuMind — available commands"
	@echo "  ──────────────────────────────"
	@echo "  make install   Install all dependencies"
	@echo "  make ui        Start Streamlit chat interface"
	@echo "  make api       Start FastAPI backend"
	@echo "  make demo      Run automated demo walkthrough"
	@echo "  make test      Run test suite"
	@echo "  make coverage  Run tests with coverage report"
	@echo "  make clean     Remove generated files"
	@echo ""

install:
	pip install -r requirements.txt

ui:
	streamlit run app.py

api:
	python main.py

demo:
	python demo.py

cli:
	python cli.py chat

test:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=src --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	@echo "Cleaned."