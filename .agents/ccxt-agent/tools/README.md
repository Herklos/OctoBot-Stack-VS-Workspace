# CCXT Agent Tools

This directory contains specialized tools for the CCXT Exchange Integration Agent. These tools help with testing, debugging, and validating websocket implementations for cryptocurrency exchanges.

### ccxt_websocket_connector_tester.py

A dedicated testing tool for OctoBot's CCXTWebsocketConnector that focuses on validating feed callbacks.

#### Features

- **Callback Tracking**: Monitors all feed callbacks (ticker, trades, orderbook, candles, etc.) to ensure they are called with valid data
- **Minimal Mocking**: Uses a minimal mock of ExchangeManager while testing real CCXTWebsocketConnector functionality
- **Data Validation**: Validates callback data structures match expected formats
- **Exchange Agnostic**: Works with any exchange supported by CCXTWebsocketConnector
- **Real-time Monitoring**: Shows live callback activity during testing

#### Usage

```bash
# Test ticker and trades callbacks for Binance
python ccxt_websocket_connector_tester.py --exchange binance --feeds ticker,trades --symbols BTC/USDT

# Test orderbook with custom duration
python ccxt_websocket_connector_tester.py --exchange kraken --feeds book --symbols XBT/USD --duration 60

# Test multiple feeds and symbols
python ccxt_websocket_connector_tester.py --exchange coinbase --feeds ticker,trades,book --symbols BTC-USD,ETH-USD
```

#### Command Line Options

- `--exchange`: Exchange name (required, e.g., binance, kraken, coinbasepro)
- `--feeds`: Comma-separated feeds to test (ticker,trades,book,candle,funding,open_interest,markets)
- `--symbols`: Comma-separated symbols to test (required)
- `--duration`: Test duration in seconds (default: 30)
- `--config`: JSON config file path
- `--sandbox`: Use sandbox/testnet mode
- `--verbose`: Enable verbose logging

#### Configuration

```json
{
  "sandbox": false,
  "additional_config": {
    "throttled_ws_updates": 1.0
  }
}
```

#### Agent Integration

```python
from ccxt_websocket_connector_tester import CCXTWebsocketConnectorTester

# Test callbacks programmatically
tester = CCXTWebsocketConnectorTester("binance", config)
results = await tester.test_feeds(["ticker", "trades"], ["BTC/USDT"], duration=30)
tester.print_results(results)
```

#### Output

The tool provides detailed output including:

- Callback invocation counts and timing
- Sample data from each callback
- Data structure validation
- Success/failure status

Example output:
```
🎯 Testing binance feeds: ticker, trades
   Symbols: BTC/USDT
   Duration: 30 seconds
🔄 Connector started, monitoring callbacks...
⏱️ 5s: 12 callback calls
⏱️ 10s: 28 callback calls

📊 CCXT WEBSOCKET CONNECTOR TEST RESULTS
✅ TEST PASSED - 45 callback calls received

📞 CALLBACK ACTIVITY:
   ticker: ✅ ACTIVE (15 calls) (2.1s - 28.5s)
   trades: ✅ ACTIVE (30 calls) (1.8s - 29.2s)

✅ DATA VALIDATION:
   ✅ ticker data structure valid
   ✅ recent_trades data structure valid
   🎉 All callback data structures are valid!
```

## Tools

A comprehensive websocket testing tool for CCXT-supported exchanges.

#### Features

- **Feed Testing**: Test concurrent websocket feeds (orderbook, trades, ticker) like OctoBot does
- **Individual Method Testing**: Test each websocket method separately with detailed diagnostics
- **Raw Websocket Testing**: Connect directly to exchange websockets for low-level debugging
- **Message Format Validation**: Check if subscription messages match exchange specifications
- **Exchange Agnostic**: Works with any CCXT-supported exchange, not just specific ones

#### Usage

```bash
# Test feeds for any exchange
python websocket_tester.py --exchange binance --test feeds

# Test individual methods with custom symbol
python websocket_tester.py --exchange kraken --test methods --symbol BTC/USD

# Run comprehensive diagnostics
python websocket_tester.py --exchange coinbase --test diagnostics

# Test raw websocket connection
python websocket_tester.py --exchange custom --test raw --ws-url wss://ws.example.com
```

#### Command Line Options

- `--exchange`: Exchange name (required)
- `--test`: Test type (feeds, methods, raw, diagnostics)
- `--symbol`: Trading symbol to test
- `--asset-id`: Asset ID for exchanges that require it
- `--ws-url`: Custom websocket URL
- `--api-key`: Exchange API key
- `--secret`: Exchange API secret
- `--config`: JSON config file path
- `--duration`: Test duration in seconds (default: 30)
- `--timeout`: Individual test timeout (default: 10.0)
- `--verbose`: Enable verbose output

#### Configuration

You can provide configuration via JSON file:

```json
{
  "api_key": "your_api_key",
  "secret": "your_secret",
  "symbol": "BTC/USDT",
  "asset_id": "asset123",
  "ws_url": "wss://custom.ws.url",
  "exchange_options": {
    "verbose": true
  }
}
```

#### Agent Integration

The agent can use this tool programmatically:

```python
from websocket_tester import ExchangeWebSocketTester

# Test feeds
tester = ExchangeWebSocketTester("binance", {"symbol": "BTC/USDT"})
success = await tester.test_feeds_octobot_style()

# Test individual methods
results = await tester.test_individual_feed_methods()

# Run diagnostics
success = await run_diagnostics_test("binance", config)
```

## Architecture

The tools are designed to be:

- **Modular**: Each tool focuses on a specific aspect of testing
- **Exchange Agnostic**: Work with any CCXT exchange without hardcoded exchange-specific logic
- **Agent-Friendly**: Easy to import and use programmatically by the CCXT agent
- **CLI-Ready**: Command-line interfaces for manual testing and debugging

## Development

When adding new tools:

1. Follow the existing patterns for exchange initialization and error handling
2. Include comprehensive CLI argument parsing
3. Add proper async/await patterns for websocket operations
4. Provide both programmatic and CLI interfaces
5. Include detailed docstrings and usage examples
6. Handle cleanup properly (exchange.close(), task cancellation)

## Dependencies

- CCXT library
- websockets library
- asyncio
- argparse
- json
- logging