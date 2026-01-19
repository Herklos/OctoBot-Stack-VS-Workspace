#!/usr/bin/env python3
"""
CCXT Websocket Connector Tester

This tool tests websocket feed callbacks for CCXT exchanges, including tentacle exchanges from OctoBot-Tentacles.

Features:
- Tests all supported feeds (ticker, trades, book, etc.) for the exchange
- Supports both standard CCXT exchanges and OctoBot tentacle exchanges
- Automatically filters to only feeds supported by the exchange
- Reports callback activity and data validation

For tentacle exchanges, the tool dynamically loads CCXT classes from ../../../OctoBot-Tentacles/Trading/Exchange/{exchange}/ccxt/

Usage:
    python ccxt_websocket_connector_tester.py --exchange <exchange> --symbols <symbols> [--duration <seconds>]

If no callbacks are received, it may indicate low activity, implementation issues, or inactive symbols.
"""

import asyncio
import sys
import os
import logging
import argparse
import json
import time
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import MagicMock
import threading

# Add paths for imports - adjust based on project structure
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../../OctoBot-Tentacles")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../OctoBot-Trading")
)

# Import OctoBot components
import octobot_trading.constants as trading_constants
import octobot_trading.enums as trading_enums
from octobot_trading.exchanges.connectors.ccxt.ccxt_websocket_connector import (
    CCXTWebsocketConnector,
)
from octobot_trading.exchanges.connectors.ccxt.ccxt_adapter import CCXTAdapter
import octobot_commons.enums as commons_enums


class TestCCXTWebsocketConnector(CCXTWebsocketConnector):
    """CCXTWebsocketConnector subclass for testing with get_name() implemented"""

    EXCHANGE_FEEDS = {
        trading_enums.WebsocketFeeds.TRADES: True,
        trading_enums.WebsocketFeeds.TICKER: True,
        trading_enums.WebsocketFeeds.CANDLE: True,
        trading_enums.WebsocketFeeds.KLINE: True,
        trading_enums.WebsocketFeeds.L1_BOOK: True,
        trading_enums.WebsocketFeeds.L2_BOOK: True,
        trading_enums.WebsocketFeeds.L3_BOOK: True,
        trading_enums.WebsocketFeeds.MARKETS: True,
        trading_enums.WebsocketFeeds.FUNDING: True,
        trading_enums.WebsocketFeeds.OPEN_INTEREST: True,
        trading_enums.WebsocketFeeds.FUTURES_INDEX: True,
        trading_enums.WebsocketFeeds.ORDERS: True,
        trading_enums.WebsocketFeeds.TRADE: True,
        trading_enums.WebsocketFeeds.LEDGER: True,
        trading_enums.WebsocketFeeds.TRANSACTIONS: True,
        trading_enums.WebsocketFeeds.PORTFOLIO: True,
        trading_enums.WebsocketFeeds.POSITION: True,
    }

    @classmethod
    def get_name(cls):
        return "test_ccxt_websocket_connector"

    def _create_client(self):
        """Override to support tentacle exchanges"""
        import ccxt.pro as ccxtpro
        import importlib
        import importlib.util
        import octobot_trading.exchanges.connectors.ccxt.ccxt_client_util as ccxt_client_util

        feed_name = self.get_feed_name()
        try:
            # Try standard ccxtpro first
            client_class = getattr(ccxtpro, feed_name)
        except AttributeError:
            # Try tentacle exchange
            try:
                tentacle_path = os.path.join(
                    os.path.dirname(__file__), "../../../OctoBot-Tentacles"
                )
                # Add paths for tentacle imports
                sys.path.insert(0, tentacle_path)
                ccxt_dir = os.path.join(
                    tentacle_path, "Trading", "Exchange", feed_name, "ccxt"
                )
                sys.path.insert(0, ccxt_dir)
                ccxt_file = os.path.join(
                    tentacle_path,
                    "Trading",
                    "Exchange",
                    feed_name,
                    "ccxt",
                    f"{feed_name}_pro.py",
                )
                spec = importlib.util.spec_from_file_location(
                    f"{feed_name}_pro", ccxt_file
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load spec for {ccxt_file}")
                tentacle_module = importlib.util.module_from_spec(spec)
                tentacle_module.__package__ = f"Trading.Exchange.{feed_name}.ccxt"
                spec.loader.exec_module(tentacle_module)
                client_class = getattr(tentacle_module, feed_name)
            except (ImportError, AttributeError, FileNotFoundError):
                raise AttributeError(f"No CCXT exchange class found for {feed_name}")

        # Use the create_client utility
        self.client, self.is_authenticated = ccxt_client_util.create_client(
            client_class,
            self.exchange_manager,
            self.logger,
            self.options,
            self.headers,
            self.additional_config,
            self._should_authenticate(),
            allow_request_counter=False,
        )

        # Configure client
        if self.exchange_manager.exchange.is_supporting_sandbox():
            ccxt_client_util.set_sandbox_mode(self, self.exchange_manager.is_sandboxed)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MinimalExchangeManager:
    """Minimal mock of ExchangeManager for testing CCXTWebsocketConnector"""

    def __init__(self, exchange_name: str, config: Dict[str, Any]):
        self.exchange_name = exchange_name
        self.config = config
        self.id = f"{exchange_name}_exchange_manager"
        self.is_sandboxed = config.get("sandbox", False)
        self.is_future = False
        self.is_margin = False
        self.exchange_only = False
        self.is_spot_only = True
        self.ignore_config = True  # For testing, ignore config checks
        self.proxy_config = MagicMock()
        self.proxy_config.aiohttp_trust_env = False
        self.proxy_config.http_proxy = None
        self.proxy_config.https_proxy = None
        self.proxy_config.socks_proxy = None
        self.proxy_config.http_proxy_callback = None
        self.proxy_config.https_proxy_callback = None
        self.proxy_config.socks_proxy_callback = None
        self.is_simulated = False
        self.logger = logging.getLogger(f"{exchange_name}_exchange_manager")
        self.without_auth = True  # For testing public feeds

        # Mock methods
        self.check_config = lambda *args: True
        self.get_exchange_credentials = lambda *args: (None, None, None, False, None)

        # Create minimal exchange mock
        self.exchange = MagicMock()
        self.exchange.authenticated.return_value = False
        self.exchange.is_supporting_sandbox.return_value = True
        self.exchange.get_exchange_current_timeframe.return_value = "1m"
        self.exchange.get_additional_connector_config.return_value = {}

        # Mock exchange_symbols_data
        self.exchange_symbols_data = MagicMock()
        mock_symbol_data = MagicMock()
        mock_symbol_data.symbol_candles = {}
        self.exchange_symbols_data.get_exchange_symbol_data.return_value = (
            mock_symbol_data
        )


class CallbackTracker:
    """Tracks callback invocations and data"""

    def __init__(self):
        self.calls: Dict[str, List[Dict[str, Any]]] = {
            "ticker": [],
            "recent_trades": [],
            "book": [],
            "candle": [],
            "funding": [],
            "open_interest": [],
            "markets": [],
            "index": [],
            "orders": [],
            "trades": [],
            "balance": [],
            "transaction": [],
        }
        self.call_counts: Dict[str, int] = {key: 0 for key in self.calls.keys()}

    def track_callback(self, callback_name: str):
        """Decorator to track callback calls"""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                self.call_counts[callback_name] += 1
                # Capture the data (first arg after self is usually the data)
                data = args[1] if len(args) > 1 else kwargs
                logger.info(
                    f"Callback {callback_name} data type: {type(data)}, data: {data}"
                )
                logger.info(f"Callback {callback_name} data: {data}")
                self.calls[callback_name].append(
                    {
                        "timestamp": time.time(),
                        "data": data,
                        "args": len(args),
                        "kwargs": list(kwargs.keys()),
                    }
                )
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of callback activity"""
        return {
            "total_calls": sum(self.call_counts.values()),
            "calls_by_type": self.call_counts.copy(),
            "first_call_times": {
                name: calls[0]["timestamp"] if calls else None
                for name, calls in self.calls.items()
            },
            "last_call_times": {
                name: calls[-1]["timestamp"] if calls else None
                for name, calls in self.calls.items()
            },
        }


class CCXTWebsocketConnectorTester:
    """Tester for CCXTWebsocketConnector callbacks"""

    def __init__(self, exchange_name: str, config: Optional[Dict[str, Any]] = None):
        self.exchange_name = exchange_name
        self.config = config or {}
        self.connector: Optional[CCXTWebsocketConnector] = None
        self.exchange_manager: Optional[MinimalExchangeManager] = None
        self.callback_tracker = CallbackTracker()

    async def setup_connector(self, feeds: List[str]) -> bool:
        """Set up the CCXTWebsocketConnector with minimal mocking"""
        try:
            # Create minimal exchange manager
            self.exchange_manager = MinimalExchangeManager(
                self.exchange_name, self.config
            )

            # Create connector
            self.connector = TestCCXTWebsocketConnector(
                config=self.config,
                exchange_manager=self.exchange_manager,
                adapter_class=CCXTAdapter,
                additional_config=self.config.get("additional_config", {}),
                websocket_name=self.exchange_name,
            )

            # Map feeds to CCXT watch keys
            feed_to_watch = {
                "TICKER": "watchTicker",
                "TRADES": "watchTrades",
                "BOOK": "watchOrderBook",
                # "CANDLE": "watchOHLCV",  # Skip for now due to min_timeframe issues
                "FUNDING": "watchFundingRate",
                "OPEN_INTEREST": "watchOpenInterest",
                "ORDERS": "watchOrders",
                "TRADE": "watchMyTrades",
                # MARKETS, INDEX, BALANCE, TRANSACTION not standard
            }

            # Set channels for supported feeds only
            self.connector.channels = [
                feed.name.lower()
                for feed in self.connector.EXCHANGE_FEEDS
                if feed.name in feed_to_watch
                and self.connector.client.has.get(feed_to_watch[feed.name], False)
            ]

            # Monkey patch callbacks to track calls
            self._patch_callbacks()

            logger.info(f"✅ CCXTWebsocketConnector set up for {self.exchange_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to set up connector: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _patch_callbacks(self):
        """Patch connector callbacks to track invocations"""
        if not self.connector:
            return

        # Patch the main callbacks
        callbacks_to_patch = [
            ("ticker", self.connector.ticker),
            ("recent_trades", self.connector.recent_trades),
            ("book", self.connector.book),
            ("candle", self.connector.candle),
            ("funding", self.connector.funding),
            ("open_interest", self.connector.open_interest),
            ("markets", self.connector.markets),
            ("index", self.connector.index),
            ("orders", self.connector.orders),
            ("trades", self.connector.trades),
            ("balance", self.connector.balance),
            ("transaction", self.connector.transaction),
        ]

        for callback_name, original_callback in callbacks_to_patch:
            setattr(
                self.connector,
                callback_name,
                self.callback_tracker.track_callback(callback_name)(original_callback),
            )

    async def subscribe_feed(self, feed_name: str, symbol: str, method_name: str):
        """Subscribe to a specific feed asynchronously in the connector's loop"""
        for feed in [feed_name]:  # wrap in list for consistency
            if feed in self.callback_tracker.calls:
                try:
                    if feed == "ticker":
                        await self._do_subscribe(symbol, "watchTicker")
                    elif feed == "trades":
                        await self._do_subscribe(symbol, "watchTrades")
                    elif feed == "book":
                        await self._do_subscribe(symbol, "watchOrderBook")
                    # Add more if needed
                    else:
                        continue
                except Exception as e:
                    logger.error(f"Failed to subscribe to {feed}: {e}")
                    raise

    async def _do_subscribe(self, symbol, method_name):
        method = getattr(self.connector.client, method_name)
        await method(symbol)
        logger.info(f"Subscribed to {method_name} for {symbol}")

    async def test_feeds(
        self, feeds: List[str], symbols: List[str], duration: int = 30
    ) -> Dict[str, Any]:
        print("Starting test_feeds")
        print(
            f"🎯 Testing {self.exchange_name} feeds: {', '.join([f for f in feeds if f in self.callback_tracker.calls])}"
        )
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Duration: {duration} seconds")

        if not self.connector:
            if not await self.setup_connector(feeds):
                return {"success": False, "error": "Connector setup failed"}

        assert self.connector is not None

        # Handle AUTO symbol
        if symbols == ["AUTO"]:
            print("Handling AUTO symbol")
            try:
                await self.connector.client.load_markets()
                if self.connector.client.symbols:
                    symbols = [self.connector.client.symbols[0]]
                    print(f"   Auto-selected symbol: {symbols[0]}")
                else:
                    return {
                        "success": False,
                        "error": "No symbols available for exchange",
                    }
            except Exception as e:
                return {"success": False, "error": f"Failed to load markets: {e}"}

        # Add pairs to connector
        self.connector.add_pairs(symbols)
        logger.info(f"Filtered pairs after add_pairs: {self.connector.filtered_pairs}")
        print("After add_pairs")

        # Subscribe to feeds directly (CCXT starts WS on watch)
        subscription_tasks = []
        for feed in feeds:
            callback_key = feed
            if feed == "trades":
                callback_key = "recent_trades"
            if callback_key in self.callback_tracker.calls:
                try:
                    print(f"About to subscribe to {feed}")
                    if feed == "ticker":
                        task = asyncio.create_task(
                            self.subscribe_feed(feed, symbols[0], "watchTicker")
                        )
                    elif feed == "trades":
                        task = asyncio.create_task(
                            self.subscribe_feed(feed, symbols[0], "watchTrades")
                        )
                    elif feed == "book":
                        task = asyncio.create_task(
                            self.subscribe_feed(feed, symbols[0], "watchOrderBook")
                        )
                    # Add more if needed
                    else:
                        continue
                    subscription_tasks.append(task)
                except Exception as e:
                    logger.error(f"Failed to subscribe to {feed}: {e}")
                    print(f"Exception: {e}")

        # Wait for subscriptions
        if subscription_tasks:
            print("Waiting for subscription tasks")
            await asyncio.gather(*subscription_tasks, return_exceptions=True)
            print("Subscription tasks done")

        print("🔄 Connector started, monitoring callbacks...")

        # Monitor for specified duration
        start_time = time.time()
        last_log_time = start_time

        while time.time() - start_time < duration:
            await asyncio.sleep(1)

            # Log progress every 5 seconds
            if time.time() - last_log_time >= 5:
                summary = self.callback_tracker.get_summary()
                total_calls = summary["total_calls"]
                print(
                    f"⏱️ {int(time.time() - start_time)}s: {total_calls} callback calls"
                )
                last_log_time = time.time()

        # Stop connector
        await self.connector.stop()
        thread.join(timeout=5)

        # Get results
        summary = self.callback_tracker.get_summary()
        results = {
            "success": summary["total_calls"] > 0,
            "duration": duration,
            "feeds_tested": [f for f in feeds if f in self.callback_tracker.calls],
            "symbols_tested": symbols,
            "callback_summary": summary,
            "detailed_calls": self.callback_tracker.calls,
        }

        return results

    def print_results(self, results: Dict[str, Any]):
        """Print test results"""
        print("\n" + "=" * 60)
        print("📊 CCXT WEBSOCKET CONNECTOR TEST RESULTS")
        print("=" * 60)

        if not results.get("success", False):
            print("❌ TEST FAILED")
            if "error" in results:
                print(f"Error: {results['error']}")
            return

        summary = results["callback_summary"]
        total_calls = summary["total_calls"]

        print(f"✅ TEST PASSED - {total_calls} callback calls received")
        print(f"   Duration: {results['duration']} seconds")
        print(
            f"   Feeds tested: {', '.join([f for f in results['feeds_tested'] if f in self.callback_tracker.calls])}"
        )
        print(f"   Symbols tested: {', '.join(results['symbols_tested'])}")

        print("\n📞 CALLBACK ACTIVITY:")
        for callback_name, count in summary["calls_by_type"].items():
            status = "✅ ACTIVE" if count > 0 else "❌ NO CALLS"
            first_time = summary["first_call_times"][callback_name]
            last_time = summary["last_call_times"][callback_name]
            timing = ""
            if first_time and last_time:
                timing = f" ({first_time:.1f}s - {last_time:.1f}s)"
            print(f"   {callback_name}: {status} ({count} calls{timing})")

        print("\n🔍 DETAILED CALLBACK DATA:")
        active_feeds = [
            name for name, calls in results["detailed_calls"].items() if calls
        ]
        if not active_feeds:
            print("   No callbacks received. This may indicate:")
            print("   - Low/no trading activity on the test symbol")
            print("   - Issue with websocket implementation or authentication")
            print("   - Symbol not found or inactive")
            return

        for callback_name, calls in results["detailed_calls"].items():
            if calls:
                print(f"   {callback_name.upper()}:")
                for i, call in enumerate(calls[:3]):  # Show first 3 calls
                    data_preview = (
                        str(call["data"])[:100] + "..."
                        if len(str(call["data"])) > 100
                        else str(call["data"])
                    )
                    print(f"     Call {i + 1}: {data_preview}")
                if len(calls) > 3:
                    print(f"     ... and {len(calls) - 3} more calls")

        # Validate data structure
        print("\n✅ DATA VALIDATION:")
        validation_passed = True

        for callback_name, calls in results["detailed_calls"].items():
            if not calls:
                continue

            sample_call = calls[0]["data"]
            adapted_data = None

            try:
                if self.connector and self.connector.adapter:
                    if callback_name == "ticker":
                        adapted_data = self.connector.adapter.adapt_ticker(sample_call)
                    elif callback_name == "book":
                        adapted_data = self.connector.adapter.adapt_order_book(
                            sample_call
                        )
                    elif callback_name == "recent_trades":
                        adapted_data = (
                            self.connector.adapter.adapt_public_recent_trades(
                                sample_call
                            )
                        )
                    # Add more if needed, e.g., candle, funding
                else:
                    adapted_data = sample_call
            except Exception as e:
                logger.warning(f"Failed to adapt {callback_name} data: {e}")
                adapted_data = sample_call  # Fall back to raw

            if callback_name == "ticker" and isinstance(adapted_data, dict):
                required_keys = ["last", "bid", "ask"]
                missing_keys = [k for k in required_keys if k not in adapted_data]
                if missing_keys:
                    print(
                        f"   ⚠️ ticker callback missing keys: {missing_keys} (adapted: {adapted_data})"
                    )
                    validation_passed = False
                else:
                    print("   ✅ ticker data structure valid")

            elif (
                callback_name == "recent_trades"
                and isinstance(adapted_data, list)
                and adapted_data
            ):
                trade = (
                    adapted_data[0]
                    if isinstance(adapted_data[0], dict)
                    else adapted_data[0]
                )
                if isinstance(trade, dict) and "price" in trade and "amount" in trade:
                    print("   ✅ recent_trades data structure valid")
                else:
                    print("   ⚠️ recent_trades data structure incomplete")
                    validation_passed = False

            elif callback_name == "book" and isinstance(adapted_data, dict):
                if "bids" in adapted_data and "asks" in adapted_data:
                    print("   ✅ book data structure valid")
                else:
                    print("   ⚠️ book data structure incomplete")
                    validation_passed = False
            else:
                # For minimal data (e.g., subscription confirmations), accept if symbol present
                if isinstance(adapted_data, dict) and "symbol" in adapted_data:
                    print(
                        f"   ✅ {callback_name} minimal data valid (subscription confirmation)"
                    )
                else:
                    print(f"   ⚠️ {callback_name} data structure unknown or incomplete")
                    validation_passed = False

        if validation_passed:
            print("   🎉 All callback data structures are valid!")
        else:
            print("   ⚠️ Some callback data structures may be incomplete")


async def run_callback_test(
    exchange_name: str,
    feeds: List[str],
    symbols: List[str],
    duration: int,
    config: Dict[str, Any],
) -> int:
    """Run the callback test"""
    tester = CCXTWebsocketConnectorTester(exchange_name, config)
    results = await tester.test_feeds(feeds, symbols, duration)
    tester.print_results(results)
    return 0 if results.get("success", False) else 1


def main():
    parser = argparse.ArgumentParser(
        description="CCXT Websocket Connector Tester - Tests feed callbacks for standard and tentacle exchanges. Only supported feeds will be tested.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   # Test ticker and trades feeds for Binance
   python ccxt_websocket_connector_tester.py --exchange binance --feeds ticker,trades --symbols BTC/USDT

   # Test all feeds with custom duration
   python ccxt_websocket_connector_tester.py --exchange kraken --feeds ticker,trades,book --symbols XBT/USD --duration 60

   # Test with config file
   python ccxt_websocket_connector_tester.py --exchange binance --feeds ticker --symbols BTC/USDT --config config.json

   # Test tentacle exchange (e.g., Polymarket) - only supported feeds will be tested
   python ccxt_websocket_connector_tester.py --exchange polymarket --symbols "will-bitcoin-replace-sha-256-before-2027/USDC:USDC-261231-0-YES" --duration 30

 Supported feeds depend on the exchange. Tentacle exchanges are loaded dynamically from OctoBot-Tentacles.
         """,
    )

    parser.add_argument(
        "--exchange",
        required=True,
        help="Exchange name (e.g., binance, kraken, coinbasepro). Supports tentacle exchanges like polymarket.",
    )
    parser.add_argument(
        "--feeds",
        help="Comma-separated list of feeds to test (ticker,trades,book,candle,funding,open_interest,markets,index,orders,trades,balance,transaction). If not specified, tests all feeds. Note: Only supported feeds for the exchange will be tested.",
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated list of symbols to test (e.g., BTC/USDT,ETH/USDT) or 'AUTO' to use the first available symbol",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds (default: 30)",
    )
    parser.add_argument("--config", help="JSON config file path")
    parser.add_argument(
        "--sandbox", action="store_true", help="Use sandbox/testnet mode"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Load config
    config = {"sandbox": args.sandbox}
    if args.config:
        try:
            with open(args.config, "r") as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return 1

    # Define valid feeds
    valid_feeds = [
        "ticker",
        "trades",
        "book",
        "candle",
        "funding",
        "open_interest",
        "markets",
        "index",
        "orders",
        "trades",  # Note: recent_trades is TRADES, trades is TRADE
        "balance",
        "transaction",
    ]

    # Parse feeds and symbols
    if args.feeds:
        feeds = [f.strip() for f in args.feeds.split(",")]
    else:
        feeds = valid_feeds.copy()  # Test all feeds
    symbols = [s.strip() for s in args.symbols.split(",")]
    invalid_feeds = [f for f in feeds if f not in valid_feeds]
    if invalid_feeds:
        print(f"❌ Invalid feeds: {invalid_feeds}")
        print(f"Valid feeds: {valid_feeds}")
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run test
    async def run():
        try:
            return await run_callback_test(
                args.exchange, feeds, symbols, args.duration, config
            )
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted")
            return 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback

            traceback.print_exc()
            return 1

    exit_code = asyncio.run(run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
