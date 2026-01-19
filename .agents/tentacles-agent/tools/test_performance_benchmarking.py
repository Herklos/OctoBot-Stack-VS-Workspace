"""
Test script for Performance Benchmarking Framework

Validates the performance benchmarking functionality by running sample benchmarks
and demonstrating baseline establishment and regression detection.
"""

import time
import sys
import os

# Add the tools directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from performance_benchmarking import PerformanceBenchmarker

    PERFORMANCE_AVAILABLE = True
except ImportError as e:
    print(f"Performance benchmarking not available: {e}")
    PERFORMANCE_AVAILABLE = False


def sample_operation_fast():
    """Fast operation for benchmarking"""
    time.sleep(0.1)
    return "fast_result"


def sample_operation_medium():
    """Medium-speed operation for benchmarking"""
    time.sleep(0.5)
    return "medium_result"


def sample_operation_slow():
    """Slow operation for benchmarking"""
    time.sleep(1.0)
    return "slow_result"


def sample_operation_regression():
    """Operation that will show regression (slower than baseline)"""
    time.sleep(2.0)  # Much slower than baseline
    return "regression_result"


def test_performance_benchmarking():
    """Test the performance benchmarking framework"""
    if not PERFORMANCE_AVAILABLE:
        print("❌ Performance benchmarking module not available")
        return False

    print("🧪 Testing Performance Benchmarking Framework")
    print("=" * 50)

    # Create benchmarker
    benchmarker = PerformanceBenchmarker("test_baselines.json")

    try:
        # Test 1: Benchmark individual operations
        print("\n📊 Benchmarking individual operations...")

        result1 = benchmarker.benchmark_operation("fast_op", sample_operation_fast)
        result2 = benchmarker.benchmark_operation("medium_op", sample_operation_medium)
        result3 = benchmarker.benchmark_operation("slow_op", sample_operation_slow)

        print(f"✅ Fast operation: {result1.duration_seconds:.3f}s")
        print(f"✅ Medium operation: {result2.duration_seconds:.3f}s")
        print(f"✅ Slow operation: {result3.duration_seconds:.3f}s")

        # Test 2: Establish baselines
        print("\n📈 Establishing performance baselines...")

        # Run multiple times for baseline
        fast_results = []
        for i in range(3):
            result = benchmarker.benchmark_operation("fast_op", sample_operation_fast)
            fast_results.append(result)

        baseline = benchmarker.establish_baseline("fast_op", fast_results)
        print(
            f"✅ Baseline established for fast_op: {baseline.baseline_duration_seconds:.3f}s"
        )

        # Test 3: Check for regression
        print("\n🔍 Testing regression detection...")

        # This should not trigger regression (within tolerance)
        normal_result = benchmarker.benchmark_operation(
            "fast_op", sample_operation_fast
        )
        regression_check = benchmarker.check_regression(normal_result)
        if regression_check:
            print(
                f"⚠️  Unexpected regression detected: {regression_check.is_regression}"
            )
        else:
            print("✅ No regression detected for normal operation")

        # This should trigger regression (much slower)
        regression_result = benchmarker.benchmark_operation(
            "fast_op", sample_operation_regression
        )
        regression_check = benchmarker.check_regression(regression_result)
        if regression_check:
            print("✅ Regression correctly detected:")
            print(
                f"   Duration regression: {regression_check.duration_regression_percent:.1f}%"
            )
            print(f"   Recommendations: {regression_check.recommendations}")
        else:
            print("❌ Regression not detected (expected)")

        # Test 4: Export results
        print("\n💾 Exporting benchmark results...")
        benchmarker.export_results("test_benchmark_results.json")
        benchmarker.export_results("test_benchmark_results.csv", format="csv")
        print("✅ Results exported to test_benchmark_results.json and .csv")

        # Test 5: Get performance summary
        print("\n📋 Performance Summary:")
        summary = benchmarker.get_performance_summary()
        print(f"   Total operations: {summary['total_operations']}")
        print(f"   Total baselines: {summary['total_baselines']}")
        print(f"   Regressions detected: {summary['regressions_detected']}")

        if summary["regressions_detected"] > 0:
            print("   Regression details:")
            for reg in summary["regressions"]:
                print(
                    f"     - {reg['operation']}: {reg['duration_regression']:.1f}% duration regression"
                )

        print("\n🎉 All performance benchmarking tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up test files
        try:
            os.remove("test_baselines.json")
            os.remove("test_benchmark_results.json")
            os.remove("test_benchmark_results.csv")
        except:
            pass


if __name__ == "__main__":
    success = test_performance_benchmarking()
    sys.exit(0 if success else 1)
