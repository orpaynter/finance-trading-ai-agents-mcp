# Repository Guidelines

## Project Structure & Module Organization

- `finance_trading_ai_agents_mcp/`: main Python package (MCP services, plugins, streaming OHLC, validators, utilities, prompt assets).
- `examples/`: runnable samples (custom MCP, clients, server run examples).
- `docs/`: project documentation content.
- `assets/`: images used by docs/README.
- Root entrypoints/templates: `main.py`, `pyproject.toml`, `env_example`, `config_example.toml`.

## Build, Test, and Development Commands

- Install (runtime deps): `pip install -r requirements.txt`
- Install (editable, for development): `pip install -e .`
- Start server (recommended): `python -m finance_trading_ai_agents_mcp serve --host 127.0.0.1 -p 11999`
- CLI entrypoint (after install): `finance-trading-ai-agents-mcp serve -p 11999`
- Generate a custom MCP template: `python -m finance_trading_ai_agents_mcp generate -o my_custom_mcp.py`
- Quick smoke run: `python -m finance_trading_ai_agents_mcp --help`

## Configuration

- Secrets and runtime flags are read from environment variables; use `.env` (template: `env_example`) or pass `--env-file PATH`.
- You can also inject env via JSON: `--env-config '{"AITRADOS_SECRET_KEY":"..."}'` or `--env-config-file config.json`.
- Common keys:
  - `AITRADOS_SECRET_KEY`: required for finance data access.
  - `ENABLE_RPC_PUBSUB_SERVICE`: enables cross-process RPC/PubSub + plugin startup.
- TOML config template: `config_example.toml` (place in your working directory, or wherever your tooling expects it).

## Coding Style & Naming Conventions

- Python 3.10+; use 4-space indentation and `snake_case` for modules/functions, `PascalCase` for classes.
- Keep changes localized: avoid refactors that mix API behavior changes with formatting-only edits.
- Prefer explicit config/env reads over hard-coded constants; keep defaults near CLI args (`finance_trading_ai_agents_mcp/mcp_cli.py`).

## Testing Guidelines

- No first-party test suite is currently included. For changes, provide a minimal repro or update an example under `examples/`.
- Validate via smoke tests: `--help`, server startup, and one MCP tool call from a client.

## Platform Notes

- Venv: Windows `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1`; Linux/macOS `python3 -m venv .venv; source .venv/bin/activate`

## Commit & Pull Request Guidelines

- Match existing history: short, imperative subjects like `add broker mcp` / `fix mt5 plugin`; use `1.` / `2.` lists when multiple related changes ship together.
- PRs should include: what changed, how to run/verify (commands + expected output/log snippet), and any docs/example updates required.
- Checklist: linked issue (if any), no secrets in diffs/logs, update `requirements.txt` if dependencies change.

## Security & Configuration Tips

- Do not commit secrets. Use `.env` (see `env_example`) and TOML config (see `config_example.toml`).
- `AITRADOS_SECRET_KEY` is required for finance data access; redact keys in logs and PR descriptions.
