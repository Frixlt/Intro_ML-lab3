.PHONY: lint format check

# === Linting & Formatting ===
lint:
	uv run ruff check .

format:
	uv run ruff check --fix .
	uv run ruff format .

# === Full check ===
check: lint
