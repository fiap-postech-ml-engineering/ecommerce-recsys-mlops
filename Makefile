#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ecommerce-recsys-mlops
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help requirements create_environment test test-cov lint lint-fix lint-fix-unsafe format format-fix format-diff format-verbose check clean init stop

.DEFAULT_GOAL := help

help:
	@echo "Comandos disponíveis:"
	@echo "  make requirements        - Instala dependências Python (uv sync)"
	@echo "  make create_environment  - Cria ambiente virtual com uv"
	@echo "  make test                - Roda testes com output verboso"
	@echo "  make test-cov            - Roda testes com cobertura (relatório HTML)"
	@echo "  make lint                - Verifica estilo do código com Ruff"
	@echo "  make lint-fix            - Corrige automaticamente issues de linting"
	@echo "  make lint-fix-unsafe     - Corrige automaticamente issues com unsafe-fixes"
	@echo "  make format              - Verifica formatação sem modificar (Ruff)"
	@echo "  make format-fix          - Formata código com Ruff"
	@echo "  make format-diff         - Mostra diferenças de formatação sem modificar"
	@echo "  make format-verbose      - Formata código com output verboso"
	@echo "  make check               - Executa lint, format e testes (sequencial)"
	@echo "  make clean               - Remove arquivos temporários"
	@echo "  make init                - Inicia API em background"
	@echo "  make stop                - Para API"

## Dependências

requirements:
	uv sync

create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Ambiente virtual criado. Ative com:"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

## Testes

test:
	python -m pytest tests/ -v --no-cov

test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

## Lint

lint:
	python -m ruff check src/ tests/

lint-fix:
	python -m ruff check src/ tests/ --fix

lint-fix-unsafe:
	python -m ruff check src/ tests/ --fix --unsafe-fixes

## Formatação

format:
	python -m ruff format --check src/ tests/

format-fix:
	python -m ruff format src/ tests/

format-diff:
	python -m ruff format --diff src/ tests/

format-verbose:
	python -m ruff format -v src/ tests/

## Checks combinados

check: lint format test
	@echo "✓ Todos os checks passaram!"

## Limpeza

clean:
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

#################################################################################
# SERVIÇOS                                                                      #
#################################################################################

init:
	uv run uvicorn src.api:app --reload
	@echo "API iniciada. Acesse em http://localhost:8000"
