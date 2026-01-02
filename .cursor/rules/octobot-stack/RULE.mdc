---
description: "OctoBot Stack development rules: dependency hierarchy, import patterns, tentacle structure, exchange connectors, and testing conventions"
globs: ["OctoBot*/**", "Async-Channel/**", "trading-backend/**", "*Private*/**", "Package-Version-Manager/**"]
alwaysApply: false
---

# OctoBot Stack Development Rules

## Dependency Hierarchy (CRITICAL)

Respect this dependency order - lower layers must NOT depend on higher layers:

1. **Core**: `OctoBot-Commons`, `Async-Channel`, `OctoBot-Trading`, `OctoBot-Evaluators`, `OctoBot-Backtesting`, `trading-backend`
2. **Extension** (tentacles): `OctoBot-Tentacles`, `Private-Tentacles`, `Business-Private-Tentacles`
3. **Application**: `OctoBot`, `OctoBot-Binary`, `OctoBot-Script`, `OctoBot-Market-Making`, `OctoBot-Prediction-Market`
4. **Tools**: `Package-Version-Manager`

Always check imports follow this hierarchy before adding new dependencies.

## Imports

When importing across repositories:
- Use absolute imports with aliases: `import octobot_trading.exchanges as exchanges`
- Follow pattern: `import octobot_{repo}.{module} as {alias}`
- Cross-repository imports allowed via PYTHONPATH
- No circular dependencies

In test files:
- Use relative imports for tentacle classes: `from ...binance import Binance`

## Tentacles

When creating or modifying tentacles:

**Required files:**
- `__init__.py` (can be empty)
- Main tentacle file (e.g., `polymarket_exchange.py`)
- `metadata.json` (NOT `metadata.yaml` - that's only for packages)

**Optional directories:**
- `resources/` - documentation, configs
- `tests/` - test files
- `config/` - configuration files

**metadata.json structure:**
```json
{
  "version": "1.2.0",
  "origin_package": "OctoBot-Default-Tentacles",
  "tentacles": ["ClassName1", "ClassName2"],
  "tentacles-requirements": []
}
```

**Organization by type:**
- Trading → `OctoBot-Tentacles/Trading/{Exchange,Mode,Supervisor}`
- Evaluator → `OctoBot-Tentacles/Evaluator/`
- Services → `OctoBot-Tentacles/Services/`
- Automation → `OctoBot-Tentacles/Automation/`
- Backtesting → `OctoBot-Tentacles/Backtesting/`

## Exchange Tentacles

When creating exchange tentacles:

**Inherit from:**
- REST exchanges: `exchanges.RestExchange`
- Connectors: `exchanges.CCXTConnector` or custom connector
- WebSocket: `exchanges.CCXTWebsocketConnector`

**Required class attributes:**
- `DESCRIPTION = ""` (class variable)
- `DEFAULT_CONNECTOR_CLASS = ConnectorClass` (for RestExchange)
- `@classmethod get_name(cls) -> str` returning lowercase exchange name

**File naming:**
- Exchange: `{exchange_name}_exchange.py` (e.g., `polymarket_exchange.py`)
- WebSocket feed: `{exchange_name}_websocket.py`
- Connector: `{exchange_name}_connector.py` or in `ccxt/` subdirectory

## Tests

When writing tests:
- Use `pytest` with `pytest.mark.asyncio` for async tests
- Import test utilities: `import octobot_commons.tests as commons_tests`
- Use relative imports for tentacle classes: `from ...binance import Binance`
- Place tests in `tests/` directory within tentacle folder
- Test files: `test_*.py` or `*_test.py`

## General Rules

- Maintain backward compatibility when possible
- Use type hints: `typing.Optional`, `typing.Dict`, etc.
- Check dependency hierarchy before adding imports
- Tests go in `tests/` directory per repository/tentacle
