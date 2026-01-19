#!/usr/bin/env python3
"""
CCXT Exchange Websocket Testing Tool

A comprehensive tool for testing websocket feeds on any CCXT-supported exchange.
Provides diagnostics, debugging, and validation of websocket connections and data feeds.

Usage:
    python websocket_tester.py --exchange polymarket --test feeds
    python websocket_tester.py --exchange binance --test diagnostics --symbol BTC/USDT
    python websocket_tester.py --exchange kraken --test raw --url wss://ws.kraken.com

Author: CCXT Agent Tools
"""

import asyncio
import sys
import os
import logging
import argparse
import json
import time
from typing import Dict, List, Any, Optional, Callable
import websockets

# Add paths for imports - adjust based on project structure
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../OctoBot-Tentacles")
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExchangeWebSocketTester:
    """Generic websocket tester for CCXT exchanges"""

    def __init__(self, exchange_name: str, config: Optional[Dict[str, Any]] = None):
        self.exchange_name = exchange_name
        self.config = config or {}
        self.exchange = None
        self.test_symbol = self.config.get("symbol", "BTC/USDT")
        self.asset_id = self.config.get("asset_id")

    async def initialize_exchange(self) -> bool:
        """Initialize the CCXT exchange instance"""
        try:
            # Dynamic import of exchange class
            module_path = f"tentacles.Trading.Exchange.{self.exchange_name}.ccxt.{self.exchange_name}_pro"
            exchange_module = __import__(module_path, fromlist=[self.exchange_name])
            exchange_class = getattr(exchange_module, self.exchange_name)

            # Create exchange instance with config
            exchange_config = {
                "apiKey": self.config.get("api_key", ""),
                "secret": self.config.get("secret", ""),
                "enableRateLimit": True,
                **self.config.get("exchange_options", {}),
            }

            self.exchange = exchange_class(exchange_config)
            print(f"✅ {self.exchange_name.title()} exchange instance created")

            # Load markets
            await self.exchange.load_markets()
            print(f"✅ Markets loaded: {len(self.exchange.markets)} markets")

            return True

        except Exception as e:
            print(f"❌ Failed to initialize {self.exchange_name}: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def find_test_symbol(self) -> tuple[str, Optional[str]]:
        """Find a suitable test symbol and asset_id for the exchange"""
        if (
            not self.exchange
            or not hasattr(self.exchange, "markets")
            or not self.exchange.markets
        ):
            print("❌ No markets available")
            return self.test_symbol or "BTC/USDT", self.asset_id

        # If specific symbol provided, use it
        if self.test_symbol and self.test_symbol in self.exchange.markets:
            symbol = self.test_symbol
            asset_id = self._extract_asset_id(symbol)
            print(f"🎯 Using specified symbol: {symbol}, asset_id: {asset_id}")
            return symbol, asset_id

        # Try to find a good test symbol
        candidates = []
        for symbol, market in self.exchange.markets.items():
            if any(base in symbol.upper() for base in ["BTC", "ETH", "USDT", "USDC"]):
                candidates.append((symbol, market))

        if candidates:
            symbol, market = candidates[0]
            asset_id = self._extract_asset_id_from_market(market)
            print(f"🎯 Using discovered symbol: {symbol}, asset_id: {asset_id}")
            return symbol, asset_id

        # Fallback
        symbol = list(self.exchange.markets.keys())[0]
        asset_id = self._extract_asset_id_from_market(self.exchange.markets[symbol])
        print(f"🎯 Using fallback symbol: {symbol}, asset_id: {asset_id}")
        return symbol, asset_id

        # Try to find a good test symbol
        candidates = []
        for symbol, market in self.exchange.markets.items():
            if any(base in symbol.upper() for base in ["BTC", "ETH", "USDT", "USDC"]):
                candidates.append((symbol, market))

        if candidates:
            symbol, market = candidates[0]
            asset_id = self._extract_asset_id_from_market(market)
            print(f"🎯 Using discovered symbol: {symbol}, asset_id: {asset_id}")
            return symbol, asset_id

        # Fallback
        symbol = list(self.exchange.markets.keys())[0]
        asset_id = self._extract_asset_id_from_market(self.exchange.markets[symbol])
        print(f"🎯 Using fallback symbol: {symbol}, asset_id: {asset_id}")
        return symbol, asset_id

    def _extract_asset_id(self, symbol: str) -> Optional[str]:
        """Extract asset_id from symbol if available in markets"""
        if (
            not self.exchange
            or not hasattr(self.exchange, "markets")
            or not self.exchange.markets
        ):
            return None
        if symbol in self.exchange.markets:
            return self._extract_asset_id_from_market(self.exchange.markets[symbol])
        return None

    def _extract_asset_id_from_market(self, market: Dict[str, Any]) -> Optional[str]:
        """Extract asset_id from market info"""
        market_info = market.get("info", {})
        # Try different possible field names for asset_id
        for field in ["asset_id", "assetId", "clobTokenIds", "token_id", "id"]:
            if field in market_info:
                value = market_info[field]
                if isinstance(value, list) and value:
                    return str(value[0])
                elif value:
                    return str(value)
        return None

    async def test_feeds_octobot_style(self, duration: int = 30) -> bool:
        """
        Test websocket feeds with instrumentation that reproduces OctoBot's
        feed subscription and callback behavior.
        """
        print("🎯 Testing websocket feeds with OctoBot-style instrumentation...")

        if not await self.initialize_exchange():
            return False

        symbol, asset_id = await self.find_test_symbol()

        # Set up data tracking (like OctoBot channels)
        received_data = {
            "orderbook": {"received": False, "updates": 0},
            "trades": {"received": False, "updates": 0},
            "ticker": {"received": False, "updates": 0},
        }

        # Define callback functions that mimic OctoBot's push_to_channel behavior
        async def orderbook_callback(orderbook):
            """Mimic OctoBot's order book callback"""
            received_data["orderbook"]["received"] = True
            received_data["orderbook"]["updates"] += 1
            bids_count = len(orderbook.get("bids", []))
            asks_count = len(orderbook.get("asks", []))
            print(
                f"📊 ORDERBOOK [{received_data['orderbook']['updates']}]: {bids_count} bids, {asks_count} asks"
            )

        async def trades_callback(trades):
            """Mimic OctoBot's trades callback"""
            received_data["trades"]["received"] = True
            received_data["trades"]["updates"] += 1
            trades_count = len(trades) if isinstance(trades, list) else 1
            print(
                f"💰 TRADES [{received_data['trades']['updates']}]: {trades_count} trades"
            )

        async def ticker_callback(ticker):
            """Mimic OctoBot's ticker callback"""
            received_data["ticker"]["received"] = True
            received_data["ticker"]["updates"] += 1
            last_price = ticker.get("last")
            print(
                f"📈 TICKER [{received_data['ticker']['updates']}]: last={last_price}"
            )

        # Start feed tasks (reproducing OctoBot's _feed_task behavior)
        print("🔄 Starting feed subscriptions...")

        if not self.exchange:
            print("❌ Exchange not initialized")
            return False

        # Feed task functions
        async def orderbook_feed_task():
            try:
                print("📊 Subscribing to order book feed...")
                while True:
                    params = {"asset_id": asset_id} if asset_id else {}
                    orderbook = await self.exchange.watch_order_book(
                        symbol, params=params
                    )
                    await orderbook_callback(orderbook)
                    await asyncio.sleep(1)  # Throttling like OctoBot
            except Exception as e:
                print(f"❌ Order book feed error: {e}")

        async def trades_feed_task():
            try:
                print("💰 Subscribing to trades feed...")
                while True:
                    params = {"asset_id": asset_id} if asset_id else {}
                    trades = await self.exchange.watch_trades(symbol, params=params)
                    await trades_callback(trades)
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ Trades feed error: {e}")

        async def ticker_feed_task():
            try:
                print("📈 Subscribing to ticker feed...")
                while True:
                    params = {"asset_id": asset_id} if asset_id else {}
                    ticker = await self.exchange.watch_ticker(symbol, params=params)
                    await ticker_callback(ticker)
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ Ticker feed error: {e}")

        # Start all feeds concurrently (like OctoBot does)
        tasks = [
            asyncio.create_task(orderbook_feed_task()),
            asyncio.create_task(trades_feed_task()),
            asyncio.create_task(ticker_feed_task()),
        ]

        print(f"⏳ Running feeds for {duration} seconds...")
        start_time = time.time()

        # Let feeds run for a period
        while time.time() - start_time < duration:
            await asyncio.sleep(1)
            # Check if all feeds have received data
            if all(feed["received"] for feed in received_data.values()):
                print("🎉 All feeds received data!")
                break

        # Cancel tasks
        for task in tasks:
            task.cancel()

        # Wait for cancellation
        await asyncio.gather(*tasks, return_exceptions=True)

        # Results
        print("\n📊 Feed Test Results:")
        for feed_name, data in received_data.items():
            status = "✅ ACTIVE" if data["received"] else "❌ NO DATA"
            updates = data["updates"]
            print(f"   {feed_name.upper()}: {status} ({updates} updates)")

        all_received = all(feed["received"] for feed in received_data.values())
        if all_received:
            print(
                f"\n🎉 SUCCESS: All {self.exchange_name} feeds are working correctly!"
            )
            print("✅ Websocket connections established")
            print("✅ Data subscription successful")
            print("✅ Callbacks receiving updates")
        else:
            missing = [
                name for name, data in received_data.items() if not data["received"]
            ]
            print(f"\n⚠️ PARTIAL SUCCESS: Missing data from {missing}")

        # Cleanup
        if self.exchange:
            await self.exchange.close()
            print("✅ Exchange cleanup complete")

        return all_received

    async def test_individual_feed_methods(
        self, timeout: float = 10.0
    ) -> Dict[str, bool]:
        """
        Test each feed method individually with detailed instrumentation
        """
        print("🔍 Testing individual feed methods...")

        if not self.exchange:
            if not await self.initialize_exchange():
                return {}

        symbol, asset_id = await self.find_test_symbol()

        results = {}

        # Test each method individually
        methods = [
            (
                "watch_order_book",
                lambda exch: exch.watch_order_book(
                    symbol, params={"asset_id": asset_id} if asset_id else {}
                ),
            ),
            (
                "watch_trades",
                lambda exch: exch.watch_trades(
                    symbol, params={"asset_id": asset_id} if asset_id else {}
                ),
            ),
            (
                "watch_ticker",
                lambda exch: exch.watch_ticker(
                    symbol, params={"asset_id": asset_id} if asset_id else {}
                ),
            ),
        ]

        for method_name, method_func in methods:
            print(f"\n🧪 Testing {method_name}...")
            try:
                # Test connection with timeout
                result = await asyncio.wait_for(
                    method_func(self.exchange), timeout=timeout
                )
                print(f"✅ {method_name}: Successfully received data")
                results[method_name] = True
            except asyncio.TimeoutError:
                print(f"⏰ {method_name}: Timed out (no data received)")
                results[method_name] = False
            except Exception as e:
                print(f"❌ {method_name}: Error - {e}")
                results[method_name] = False

        if self.exchange:
            await self.exchange.close()

        print("\n🔍 Individual Method Results:")
        for method, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {method}: {status}")

        return results

    async def test_raw_websocket(
        self, ws_url: Optional[str] = None, subscription_msg: Optional[Dict] = None
    ) -> bool:
        """
        Test raw websocket connection to debug message flow
        """
        print("🔍 Testing raw websocket connection...")

        # Get websocket URL from exchange or use provided
        uri = ws_url
        if not uri and self.exchange:
            describe_result = self.exchange.describe()
            urls = describe_result.get("urls", {}).get("api", {}).get("ws", {})
            uri = urls.get("market") or urls.get("public") or urls.get("ws")

        if not uri:
            print("❌ No websocket URL available")
            return False

        print(f"🌐 Connecting to: {uri}")

        try:
            async with websockets.connect(uri) as websocket:
                print("✅ Connected to websocket")

                # Send subscription message if provided
                if subscription_msg:
                    print(f"📤 Sending subscription: {subscription_msg}")
                    await websocket.send(json.dumps(subscription_msg))

                print("⏳ Listening for messages...")

                message_count = 0
                start_time = time.time()

                while message_count < 10 and (time.time() - start_time) < 30:
                    try:
                        # Receive message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        message_count += 1

                        print(f"\n📨 Message {message_count}:")
                        print(f"Raw: {message}")

                        try:
                            parsed = json.loads(message)
                            print(f"Parsed: {json.dumps(parsed, indent=2)}")

                            # Check for common event types
                            if isinstance(parsed, dict):
                                self._analyze_message(parsed)

                        except json.JSONDecodeError:
                            print("Message is not valid JSON")

                    except asyncio.TimeoutError:
                        print("⏰ Timeout waiting for message")
                        break

                print(f"\n📊 Received {message_count} messages in total")
                return message_count > 0

        except Exception as e:
            print(f"❌ Raw websocket test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _analyze_message(self, message: Dict[str, Any]):
        """Analyze websocket message for common patterns"""
        event_type = (
            message.get("event_type") or message.get("event") or message.get("type")
        )
        if event_type:
            print(f"Event type: {event_type}")

            # Check for common event types
            if event_type in ["book", "orderbook", "depth"]:
                print("📊 Order book update detected!")
            elif event_type in ["trade", "trades"]:
                print("💰 Trade update detected!")
            elif event_type in ["ticker", "price_change", "last_trade_price"]:
                print("📈 Price/ticker update detected!")
            else:
                print(f"Unknown event type: {event_type}")

    async def check_message_format(self) -> bool:
        """
        Check if the message format matches expected patterns
        """
        print("📋 Checking message format...")

        if not self.exchange:
            if not await self.initialize_exchange():
                return False

        try:
            describe_result = self.exchange.describe()
            urls = describe_result.get("urls", {}).get("api", {}).get("ws", {})
            market_url = urls.get("market")
            channel_type = describe_result.get("options", {}).get("wsMarketChannelType")

            print(f"\nCCXT {self.exchange_name} configuration:")
            print(f"Market URL: {market_url}")
            print(f"Channel type: {channel_type}")
            print(f"Headers: {describe_result.get('options', {}).get('headers', {})}")

            # Check what message would be sent
            symbol, asset_id = await self.find_test_symbol()

            if asset_id and channel_type:
                message = {"type": channel_type, "assets": [asset_id]}
                print(f"\nCCXT would send message: {json.dumps(message, indent=2)}")
                return True
            else:
                print("⚠️ Could not determine message format")
                return False

        except Exception as e:
            print(f"❌ Message format check failed: {e}")
            return False
        finally:
            if self.exchange:
                await self.exchange.close()


async def run_feeds_test(exchange_name: str, config: Dict[str, Any]) -> bool:
    """Run OctoBot-style feeds test"""
    tester = ExchangeWebSocketTester(exchange_name, config)
    return await tester.test_feeds_octobot_style()


async def run_individual_methods_test(
    exchange_name: str, config: Dict[str, Any]
) -> Dict[str, bool]:
    """Run individual feed methods test"""
    tester = ExchangeWebSocketTester(exchange_name, config)
    return await tester.test_individual_feed_methods()


async def run_raw_websocket_test(exchange_name: str, config: Dict[str, Any]) -> bool:
    """Run raw websocket test"""
    tester = ExchangeWebSocketTester(exchange_name, config)
    return await tester.test_raw_websocket(
        ws_url=config.get("ws_url"), subscription_msg=config.get("subscription_msg")
    )


async def run_diagnostics_test(exchange_name: str, config: Dict[str, Any]) -> bool:
    """Run comprehensive diagnostics"""
    print(f"🔧 Running {exchange_name} Websocket Diagnostics")
    print("=" * 60)

    tester = ExchangeWebSocketTester(exchange_name, config)

    # Test 1: Raw websocket
    raw_test = await tester.test_raw_websocket()

    # Test 2: Message format check
    format_test = await tester.check_message_format()

    # Test 3: Individual methods
    individual_results = await tester.test_individual_feed_methods()
    individual_test = all(individual_results.values())

    print("\n" + "=" * 60)
    print("📊 FINAL DIAGNOSTICS RESULTS:")
    print(f"   Raw websocket: {'✅ PASSED' if raw_test else '❌ FAILED'}")
    print(f"   Message format: {'✅ PASSED' if format_test else '❌ FAILED'}")
    print(f"   Individual methods: {'✅ PASSED' if individual_test else '❌ FAILED'}")

    overall_success = raw_test and format_test and individual_test
    print(
        f"\n🎯 OVERALL STATUS: {'✅ SUCCESS' if overall_success else '❌ ISSUES FOUND'}"
    )

    if overall_success:
        print(f"✅ All {exchange_name} websocket functionality is working correctly!")
    else:
        print("⚠️ Some issues detected - check logs above")

    return overall_success


def main():
    parser = argparse.ArgumentParser(
        description="CCXT Exchange Websocket Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test feeds for Polymarket
  python websocket_tester.py --exchange polymarket --test feeds

  # Test individual methods for Binance with specific symbol
  python websocket_tester.py --exchange binance --test methods --symbol BTC/USDT

  # Run diagnostics with custom config
  python websocket_tester.py --exchange kraken --test diagnostics --config config.json

  # Test raw websocket connection
  python websocket_tester.py --exchange custom --test raw --ws-url wss://example.com/ws
        """,
    )

    parser.add_argument(
        "--exchange",
        required=True,
        help="Exchange name (e.g., polymarket, binance, kraken)",
    )
    parser.add_argument(
        "--test",
        choices=["feeds", "methods", "raw", "diagnostics"],
        help="Test type to run",
    )
    parser.add_argument("--symbol", help="Trading symbol to test (e.g., BTC/USDT)")
    parser.add_argument("--asset-id", help="Asset ID for exchanges that require it")
    parser.add_argument("--ws-url", help="Custom websocket URL for raw testing")
    parser.add_argument("--api-key", help="Exchange API key")
    parser.add_argument("--secret", help="Exchange API secret")
    parser.add_argument("--config", help="JSON config file path")
    parser.add_argument(
        "--duration", type=int, default=30, help="Test duration in seconds"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Individual test timeout"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if not args.test:
        parser.print_help()
        return

    # Load config from file if provided
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config file: {e}")
            return

    # Override config with command line args
    config.update(
        {
            "symbol": args.symbol,
            "asset_id": args.asset_id,
            "api_key": args.api_key,
            "secret": args.secret,
            "ws_url": args.ws_url,
            "duration": args.duration,
            "timeout": args.timeout,
        }
    )

    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run the appropriate test
    async def run_test():
        try:
            if args.test == "feeds":
                success = await run_feeds_test(args.exchange, config)
            elif args.test == "methods":
                results = await run_individual_methods_test(args.exchange, config)
                success = all(results.values())
            elif args.test == "raw":
                success = await run_raw_websocket_test(args.exchange, config)
            elif args.test == "diagnostics":
                success = await run_diagnostics_test(args.exchange, config)
            else:
                print(f"❌ Unknown test type: {args.test}")
                return 1

            return 0 if success else 1

        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback

            traceback.print_exc()
            return 1

    exit_code = asyncio.run(run_test())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
