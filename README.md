# OctoBot-Stack - VSCode / Cursor workspace

## Setup

```bash
python3.10 -m venv venv
git submodule update --init --recursive --force
ln -s (pwd)/OctoBot-Tentacles/ OctoBot/tentacles
ln -s (pwd)/OctoBot-Tentacles/ OctoBot-Trading/tentacles
```

Note: Don't forget to pip install requirements.txt and full_requirements.txt before using any package.

## Dev Tips

### CCXT Exchange development debug

#### Debug fetch payload and response

- Add a breakpoint at ccxt/async_support/base/exchange.py:fetch:raw_headers (search `# CIMultiDictProxy`)
