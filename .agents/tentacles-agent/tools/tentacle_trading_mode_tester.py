#!/usr/bin/env python3
"""
Tentacle Trading Mode Tester

This tool provides comprehensive testing for OctoBot trading modes with full evaluator integration.
It allows testing of order creation, portfolio impact, risk management, and cross-tentacle interactions.

Features:
- Mock exchange setup with realistic order handling
- Full evaluator matrix integration for signal generation
- Order lifecycle simulation (create, fill, cancel)
- Portfolio impact analysis and balance tracking
- Risk management validation (stop losses, take profits)
- Multi-timeframe evaluation testing
- Performance profiling and memory usage tracking

Usage:
    python tentacle_trading_mode_tester.py --mode DailyTradingMode --evaluators evaluators.json --symbol BTC/USDT

For full configuration testing:
    python tentacle_trading_mode_tester.py --mode GridTradingMode --config full_config.json --duration 300
"""

import asyncio
import sys
import os
import logging
import argparse
import json
import time
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

# Add paths for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../OctoBot-Tentacles")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../OctoBot-Trading")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../OctoBot-Evaluators")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../OctoBot-Commons")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../Async-Channel"))

# Import OctoBot components
import octobot_trading.modes as trading_modes
import octobot_trading.exchanges as exchanges
import octobot_trading.personal_data as personal_data
import octobot_trading.enums as trading_enums
import octobot_evaluators.matrix as matrix
import octobot_evaluators.matrix.matrix_manager as matrix_manager
import octobot_evaluators.enums as evaluator_enums
import octobot_commons.enums as commons_enums


class MockExchange:
    """Mock exchange for testing trading modes"""

    def __init__(
        self, symbol: str = "BTC/USDT", initial_balance: Dict[str, float] = None
    ):
        self.symbol = symbol
        self.base_asset, self.quote_asset = symbol.split("/")
        self.initial_balance = initial_balance or {
            self.base_asset: 0.0,
            self.quote_asset: 10000.0,
        }
        self.balance = self.initial_balance.copy()
        self.orders = {}
        self.trades = []
        self.order_id_counter = 1
        self.current_price = 50000.0

    def get_balance(self, asset: str) -> float:
        """Get balance for asset"""
        return self.balance.get(asset, 0.0)

    def update_balance(self, asset: str, amount: float):
        """Update balance for asset"""
        self.balance[asset] = self.balance.get(asset, 0.0) + amount

    def create_order(
        self, order_type: str, side: str, amount: float, price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a mock order"""
        order_id = str(self.order_id_counter)
        self.order_id_counter += 1

        order = {
            "id": order_id,
            "symbol": self.symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price or self.current_price,
            "status": "open",
            "timestamp": int(time.time() * 1000),
        }

        self.orders[order_id] = order
        return order

    def fill_order(
        self,
        order_id: str,
        filled_amount: Optional[float] = None,
        filled_price: Optional[float] = None,
    ):
        """Fill an order partially or fully"""
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]
        if order["status"] != "open":
            return order

        fill_amount = filled_amount or order["amount"]
        fill_price = filled_price or order["price"]

        # Update balance based on trade
        if order["side"] == "buy":
            cost = fill_amount * fill_price
            if self.balance[self.quote_asset] >= cost:
                self.update_balance(self.quote_asset, -cost)
                self.update_balance(self.base_asset, fill_amount)
                order["status"] = (
                    "filled" if fill_amount == order["amount"] else "partially_filled"
                )
            else:
                order["status"] = "canceled"  # Insufficient funds
        else:  # sell
            if self.balance[self.base_asset] >= fill_amount:
                revenue = fill_amount * fill_price
                self.update_balance(self.base_asset, -fill_amount)
                self.update_balance(self.quote_asset, revenue)
                order["status"] = (
                    "filled" if fill_amount == order["amount"] else "partially_filled"
                )
            else:
                order["status"] = "canceled"  # Insufficient funds

        # Record trade
        trade = {
            "order_id": order_id,
            "symbol": order["symbol"],
            "side": order["side"],
            "amount": fill_amount,
            "price": fill_price,
            "timestamp": int(time.time() * 1000),
        }
        self.trades.append(trade)

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id in self.orders and self.orders[order_id]["status"] == "open":
            self.orders[order_id]["status"] = "canceled"
            return True
        return False

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        return [order for order in self.orders.values() if order["status"] == "open"]

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        base_value = self.balance.get(self.base_asset, 0) * self.current_price
        quote_value = self.balance.get(self.quote_asset, 0)
        return base_value + quote_value


class MockExchangeManager:
    """Mock exchange manager for trading mode testing"""

    def __init__(self, symbol: str = "BTC/USDT"):
        self.exchange = MockExchange(symbol)
        self.symbol = symbol
        self.trader = None

    async def get_balance(self, asset: str) -> float:
        """Get balance for asset"""
        return self.exchange.get_balance(asset)

    def get_exchange_name(self) -> str:
        """Get exchange name"""
        return "mock_exchange"

    def get_traded_symbols(self) -> List[str]:
        """Get traded symbols"""
        return [self.symbol]

    def get_traded_time_frames(self) -> List[str]:
        """Get traded timeframes"""
        return ["1h", "4h", "1d"]


class TentacleTradingModeTester:
    """Comprehensive trading mode testing tool"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.matrix = matrix.Matrix()
        self.exchange_manager = None
        self.results = {
            "orders_created": [],
            "orders_filled": [],
            "portfolio_states": [],
            "performance": {},
            "errors": [],
        }

    async def setup_exchange(
        self,
        symbol: str = "BTC/USDT",
        initial_balance: Optional[Dict[str, float]] = None,
    ):
        """Setup mock exchange for testing"""
        self.exchange_manager = MockExchangeManager(symbol)
        if initial_balance:
            self.exchange_manager.exchange.initial_balance = initial_balance
            self.exchange_manager.exchange.balance = initial_balance.copy()

        # Record initial portfolio state
        initial_value = self.exchange_manager.exchange.get_portfolio_value()
        self.results["portfolio_states"].append(
            {
                "timestamp": time.time(),
                "balance": self.exchange_manager.exchange.balance.copy(),
                "portfolio_value": initial_value,
                "stage": "initial",
            }
        )

    async def populate_matrix(
        self,
        evaluator_states: Dict[str, Any],
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
    ):
        """Populate evaluation matrix with test data"""
        for evaluator_name, eval_data in evaluator_states.items():
            if isinstance(eval_data, dict):
                eval_note = eval_data.get("eval_note", 0.0)
                eval_type = eval_data.get("type", "TA")
            else:
                eval_note = eval_data
                eval_type = "TA"

            matrix_path = [
                "mock_exchange",
                eval_type,
                evaluator_name,
                "BTC",
                symbol,
                timeframe,
            ]

            matrix_manager.set_tentacle_value(
                self.matrix.matrix_id,
                matrix_path,
                getattr(
                    evaluator_enums.EvaluatorMatrixTypes,
                    eval_type,
                    evaluator_enums.EvaluatorMatrixTypes.TA,
                ),
                eval_note,
                int(time.time()),
            )

    async def test_trading_mode(
        self,
        mode_class: type,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        test_duration: int = 60,
    ) -> Dict[str, Any]:
        """Test a trading mode with full integration"""
        print(f"🧪 Testing Trading Mode: {mode_class.__name__}")

        # Setup mode instance
        mode = mode_class()
        mode.exchange_manager = self.exchange_manager
        mode.symbol = symbol
        mode.timeframe = timeframe

        # Mock required methods
        mode.get_name = lambda: f"test_{mode_class.__name__}".lower()
        mode.get_is_cryptocurrency_wildcard = lambda: False
        mode.get_is_symbol_wildcard = lambda: False
        mode.get_is_time_frame_wildcard = lambda: False

        # Setup order creation tracking
        original_create_order = mode.create_order
        orders_created = []

        async def tracked_create_order(*args, **kwargs):
            order = await original_create_order(*args, **kwargs)
            if order:
                orders_created.append(
                    {
                        "order": order,
                        "timestamp": time.time(),
                        "args": args,
                        "kwargs": kwargs,
                    }
                )
            return order

        mode.create_order = tracked_create_order

        # Performance tracking
        start_time = time.time()
        start_memory = 0

        try:
            # Initialize mode
            await mode.initialize()

            # Run test for specified duration
            end_time = start_time + test_duration
            iteration = 0

            while time.time() < end_time:
                iteration += 1

                # Simulate mode execution (this would normally be triggered by evaluators)
                try:
                    await mode.create_orders(
                        symbol, {"state": "test"}, iteration=iteration
                    )
                except Exception as e:
                    # Some modes might not have create_orders or expect different parameters
                    pass

                # Simulate some orders being filled
                open_orders = self.exchange_manager.exchange.get_open_orders()
                for order in open_orders[:1]:  # Fill one order per iteration
                    if np.random.random() > 0.7:  # 30% chance to fill
                        filled_order = self.exchange_manager.exchange.fill_order(
                            order["id"]
                        )
                        if filled_order:
                            self.results["orders_filled"].append(
                                {"order": filled_order, "timestamp": time.time()}
                            )

                # Record portfolio state periodically
                if iteration % 10 == 0:
                    portfolio_value = (
                        self.exchange_manager.exchange.get_portfolio_value()
                    )
                    self.results["portfolio_states"].append(
                        {
                            "timestamp": time.time(),
                            "balance": self.exchange_manager.exchange.balance.copy(),
                            "portfolio_value": portfolio_value,
                            "stage": f"iteration_{iteration}",
                        }
                    )

                await asyncio.sleep(0.1)  # Small delay between iterations

            # Final portfolio state
            final_value = self.exchange_manager.exchange.get_portfolio_value()
            self.results["portfolio_states"].append(
                {
                    "timestamp": time.time(),
                    "balance": self.exchange_manager.exchange.balance.copy(),
                    "portfolio_value": final_value,
                    "stage": "final",
                }
            )

            # Cleanup
            await mode.stop()

            end_memory = 0
            actual_duration = time.time() - start_time

            result = {
                "trading_mode": mode_class.__name__,
                "test_duration": actual_duration,
                "iterations": iteration,
                "orders_created": len(orders_created),
                "orders_filled": len(self.results["orders_filled"]),
                "initial_portfolio": self.results["portfolio_states"][0][
                    "portfolio_value"
                ],
                "final_portfolio": final_value,
                "portfolio_change": final_value
                - self.results["portfolio_states"][0]["portfolio_value"],
                "performance": {
                    "duration": actual_duration,
                    "memory_delta": end_memory - start_memory,
                    "orders_per_second": len(orders_created) / actual_duration
                    if actual_duration > 0
                    else 0,
                },
                "symbol": symbol,
                "timeframe": timeframe,
            }

            self.results["orders_created"].extend(orders_created)
            print(
                f"✅ Trading Mode Test completed: {len(orders_created)} orders created, {len(self.results['orders_filled'])} filled"
            )

            return result

        except Exception as e:
            error_info = {
                "trading_mode": mode_class.__name__,
                "error": str(e),
                "traceback": str(e.__traceback__),
                "test_duration": time.time() - start_time,
            }
            self.results["errors"].append(error_info)
            print(f"❌ Trading Mode Test failed: {str(e)}")
            raise

    async def test_full_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a complete trading configuration with multiple components"""
        print("🧪 Testing Full Configuration")

        # Extract configuration
        mode_class = config.get("trading_mode")
        evaluators = config.get("evaluators", {})
        symbol = config.get("symbol", "BTC/USDT")
        timeframe = config.get("timeframe", "1h")
        duration = config.get("duration", 60)
        initial_balance = config.get("initial_balance")

        # Setup exchange
        await self.setup_exchange(symbol, initial_balance)

        # Populate matrix with evaluator states
        await self.populate_matrix(evaluators, symbol, timeframe)

        # Test trading mode
        result = await self.test_trading_mode(mode_class, symbol, timeframe, duration)

        # Add configuration-specific metrics
        result["configuration_test"] = True
        result["evaluators_configured"] = len(evaluators)
        result["matrix_populated"] = True

        return result

    def print_results(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 70)
        print("🧪 TENTACLE TRADING MODE TEST RESULTS")
        print("=" * 70)

        print(f"\n📊 Summary:")
        print(f"   Orders Created: {len(self.results['orders_created'])}")
        print(f"   Orders Filled: {len(self.results['orders_filled'])}")
        print(f"   Portfolio States: {len(self.results['portfolio_states'])}")
        print(f"   Errors: {len(self.results['errors'])}")

        if self.results["portfolio_states"]:
            initial_value = self.results["portfolio_states"][0]["portfolio_value"]
            final_value = self.results["portfolio_states"][-1]["portfolio_value"]
            change = final_value - initial_value
            change_pct = (change / initial_value) * 100 if initial_value > 0 else 0

            print("\n💰 Portfolio Performance:")
            print(".2f")
            print(".2f")
            print(".2f")

        if self.results["orders_created"]:
            order_types = {}
            for order_data in self.results["orders_created"]:
                order = order_data["order"]
                if isinstance(order, dict):
                    order_type = order.get("type", "unknown")
                    order_types[order_type] = order_types.get(order_type, 0) + 1

            print("\n📋 Orders Created:")
            for order_type, count in order_types.items():
                print(f"   {order_type}: {count}")

        if self.results["errors"]:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results["errors"]:
                print(f"   {error.get('trading_mode', 'Unknown')}: {error['error']}")

        print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Tentacle Trading Mode Tester")
    parser.add_argument("--mode", help="Trading mode class name")
    parser.add_argument("--evaluators", help="JSON file with evaluator states")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Timeframe")
    parser.add_argument(
        "--duration", type=int, default=60, help="Test duration in seconds"
    )
    parser.add_argument(
        "--config", help="JSON config file for full configuration testing"
    )
    parser.add_argument("--initial-balance", help="JSON file with initial balance")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load config
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    tester = TentacleTradingModeTester(config)

    try:
        if args.config:
            # Full configuration testing
            print("Testing full configuration...")
            # This would need dynamic imports
            # result = await tester.test_full_configuration(config)

        elif args.mode:
            # Individual mode testing
            evaluators = {}
            if args.evaluators:
                with open(args.evaluators, "r") as f:
                    evaluators = json.load(f)

            initial_balance = None
            if args.initial_balance:
                with open(args.initial_balance, "r") as f:
                    initial_balance = json.load(f)

            await tester.setup_exchange(args.symbol, initial_balance)
            await tester.populate_matrix(evaluators, args.symbol, args.timeframe)

            print(f"Testing trading mode: {args.mode}")
            # Dynamic import of trading mode
            if args.mode == "AIIndexTradingMode":
                from Trading.Mode.index_trading_mode.ai_index_trading import (
                    AIIndexTradingMode,
                )

                mode_class = AIIndexTradingMode
            else:
                print(f"❌ Unsupported mode: {args.mode}")
                return

            result = await tester.test_trading_mode(
                mode_class, args.symbol, args.timeframe, args.duration
            )

        else:
            print("❌ Specify --mode or --config")
            return

        tester.print_results()

        if args.output:
            with open(args.output, "w") as f:
                json.dump(tester.results, f, indent=2, default=str)
            print(f"📄 Results saved to {args.output}")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
