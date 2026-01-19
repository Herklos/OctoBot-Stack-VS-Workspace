# Tentacles Agent Tools

This directory contains specialized testing and instrumentation tools for the OctoBot Tentacles Agent. These tools enable deep testing of evaluators, trading modes, and complete configurations to ensure tentacles work correctly within the OctoBot ecosystem.

## Tools Overview

### tentacle_evaluator_tester.py

**Purpose**: Comprehensive testing of individual evaluators with mock data and matrix validation.

**Key Features**:
- Mock OHLCV data generation with realistic price movements
- TA evaluator testing with configurable candle counts
- Strategy evaluator testing with pre-populated TA states
- Social evaluator testing with simulated service feeds
- Matrix state validation and pertinence checking
- Performance benchmarking and memory profiling

**Usage Examples**:
```bash
# Test TA evaluator
python tentacle_evaluator_tester.py --ta RSIEvaluator --symbol BTC/USDT --timeframe 1h --candle-count 200

# Test strategy evaluator with TA dependencies
python tentacle_evaluator_tester.py --strategy SimpleStrategyEvaluator --ta-states ta_states.json --symbol BTC/USDT

# Test AI-powered strategy evaluators
python tentacle_evaluator_tester.py --strategy LLMAIStrategyEvaluator --ta-states ta_states.json --symbol BTC/USDT
python tentacle_evaluator_tester.py --strategy SentimentLLMAIStrategyEvaluator --ta-states social_states.json --symbol BTC/USDT
python tentacle_evaluator_tester.py --strategy TechnicalLLMAIStrategyEvaluator --ta-states ta_states.json --symbol BTC/USDT

# Test social evaluator with feed data
python tentacle_evaluator_tester.py --social TwitterSentimentEvaluator --feed-data twitter_feed.json
```

### tentacle_trading_mode_tester.py

**Purpose**: Integration testing of trading modes with full evaluator and exchange simulation.

**Key Features**:
- Mock exchange setup with realistic order handling
- Full evaluator matrix integration for signal generation
- Order lifecycle simulation (create, fill, cancel)
- Portfolio impact analysis and balance tracking
- Risk management validation (stop losses, take profits)
- Multi-timeframe evaluation testing
- Performance profiling

**Usage Examples**:
```bash
# Test trading mode with evaluators
python tentacle_trading_mode_tester.py --mode DailyTradingMode --evaluators evaluators.json --symbol BTC/USDT --duration 120

# Test with custom initial balance
python tentacle_trading_mode_tester.py --mode GridTradingMode --initial-balance balance.json --duration 300

# Test AI Index Trading Mode with AI evaluators
python tentacle_trading_mode_tester.py --mode AIIndexTradingMode --evaluators ai_evaluators.json --symbol BTC/USDT --duration 120
```

### tentacle_configuration_tester.py

**Purpose**: End-to-end testing of complete tentacle configurations and profiles.

**Key Features**:
- Complete profile activation with all tentacles loaded
- Cross-tentacle interaction validation
- Memory usage and performance profiling
- Error handling and recovery scenarios
- Backtesting integration for historical validation
- Multi-timeframe and multi-symbol testing

**Usage Examples**:
```bash
# Test profile configuration
python tentacle_configuration_tester.py --profile daily_trading --duration 300

# Test custom configuration with backtest data
python tentacle_configuration_tester.py --config custom_config.json --backtest-data historical.json --validate-interactions

# Test with exchange configurations
python tentacle_configuration_tester.py --profile arbitrage_trading --exchanges exchanges.json --test-recovery
```

## Data Formats

### TA States JSON (for strategy testing)
```json
{
  "RSIEvaluator": 0.8,
  "MACDEvaluator": -0.3,
  "MovingAverageEvaluator": 0.2
}
```

### Feed Data JSON (for social testing)
```json
[
  {
    "text": "Bitcoin is mooning! 🚀",
    "sentiment": "positive",
    "timestamp": 1640995200000
  }
]
```

### Evaluator Configuration JSON
```json
{
  "RSIEvaluator": {
    "eval_note": 0.75,
    "type": "TA"
  },
  "SimpleStrategyEvaluator": {
    "eval_note": 0.6,
    "type": "STRATEGIES"
  }
}
```

### Initial Balance JSON
```json
{
  "BTC": 0.5,
  "USDT": 10000.0
}
```

### Custom Configuration JSON
```json
{
  "trading_mode": "DailyTradingMode",
  "evaluators": {
    "RSIEvaluator": 0.8,
    "MACDEvaluator": -0.2
  },
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "duration": 300,
  "initial_balance": {
    "BTC": 0.0,
    "USDT": 10000.0
  }
}
```

## Agent Integration

The tentacles agent can use these tools programmatically for automated testing:

```python
from tentacle_evaluator_tester import TentacleEvaluatorTester
from tentacle_trading_mode_tester import TentacleTradingModeTester
from tentacle_configuration_tester import TentacleConfigurationTester

# Test evaluator
evaluator_tester = TentacleEvaluatorTester()
result = await evaluator_tester.test_ta_evaluator(RSIEvaluator, "BTC/USDT", "1h", 100)

# Test trading mode
trading_tester = TentacleTradingModeTester()
await trading_tester.setup_exchange("BTC/USDT")
result = await trading_tester.test_trading_mode(DailyTradingMode, "BTC/USDT", "1h", 60)

# Test configuration
config_tester = TentacleConfigurationTester()
await config_tester.load_profile("daily_trading")
results = await config_tester.run_configuration_test(300)
```

## Dependencies

All tools require:
- **OctoBot Stack**: Proper PYTHONPATH setup to access OctoBot modules
- **numpy**: For realistic data generation and statistical operations
- **psutil**: For performance profiling (optional, used in configuration tester)

## Setup

Ensure OctoBot directories are in PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/OctoBot-Tentacles:$(pwd)/OctoBot-Evaluators:$(pwd)/OctoBot-Trading:$(pwd)/OctoBot-Commons:$(pwd)/Async-Channel"
```

## Output and Results

All tools provide:
- **Console Output**: Real-time progress with emojis and status indicators
- **JSON Output**: Detailed results saved to files with `--output` parameter
- **Performance Metrics**: CPU, memory, and timing statistics
- **Error Reporting**: Comprehensive error handling with stack traces

## Development Notes

When extending these tools:
1. Maintain async/await patterns for all OctoBot interactions
2. Use proper mock objects to avoid external dependencies
3. Include comprehensive error handling and validation
4. Follow the existing naming and structure conventions
5. Add new data formats as needed for different tentacle types
6. Update documentation with new features and examples

## Limitations

- **Dynamic Imports**: Tools expect tentacle classes to be importable
- **Mock Accuracy**: Mock exchanges and data may not perfectly replicate real conditions
- **Performance**: Large-scale testing may require significant resources
- **Real-time Features**: Tools focus on isolated testing, not live trading simulation

## Troubleshooting

### Common Issues

1. **Import Errors**: Verify PYTHONPATH includes all OctoBot directories
2. **Matrix Errors**: Ensure OctoBot-Evaluators is properly initialized
3. **Memory Issues**: Reduce test duration or data sizes for large configurations
4. **Async Errors**: Check that all await calls are properly handled

### Debug Mode

Enable verbose logging for detailed operation information:

```bash
python tentacle_*.py --verbose
```

## Related Documentation

- [OctoBot Stack Architecture](../../.claude/skills/octobot-stack/)
- [Testing Python Patterns](../../.claude/skills/testing-python/)
- [CCXT Agent Tools](../ccxt-agent/tools/) (similar patterns)