"""
Automated Reporting System for OctoBot Tentacles Agent

Generates comprehensive test reports with analytics, CI/CD integration,
and multiple output formats (HTML, JSON, Markdown, PDF).
"""

import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import base64
import hashlib

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from parallel_execution_framework import ExecutionMode

try:
    from performance_benchmarking import PerformanceBenchmarker

    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False


@dataclass
class TestResult:
    """Represents a single test result"""

    test_id: str
    test_name: str
    test_type: str  # 'ta', 'strategy', 'social', 'trading_mode', 'integration'
    status: str  # 'passed', 'failed', 'error', 'skipped'
    duration: float
    timestamp: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestSuiteResult:
    """Represents a complete test suite execution"""

    suite_id: str
    suite_name: str
    execution_mode: ExecutionMode
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0
    total_duration: float = 0.0
    test_results: List[TestResult] = None
    resource_usage: Dict[str, Any] = None
    recommendations: List[str] = None
    performance_summary: Optional[Dict[str, Any]] = (
        None  # Performance benchmarking results
    )

    def __post_init__(self):
        if self.test_results is None:
            self.test_results = []
        if self.resource_usage is None:
            self.resource_usage = {}
        if self.recommendations is None:
            self.recommendations = []
        if self.performance_summary is None:
            self.performance_summary = {}

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def average_test_duration(self) -> float:
        """Calculate average test duration"""
        if not self.test_results:
            return 0.0
        return sum(result.duration for result in self.test_results) / len(
            self.test_results
        )

    def add_test_result(self, result: TestResult):
        """Add a test result and update counters"""
        self.test_results.append(result)
        self.total_tests += 1

        if result.status == "passed":
            self.passed_tests += 1
        elif result.status == "failed":
            self.failed_tests += 1
        elif result.status == "error":
            self.error_tests += 1
        elif result.status == "skipped":
            self.skipped_tests += 1

    def finalize(self):
        """Finalize the test suite with end time and duration"""
        self.end_time = datetime.now()
        if self.start_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()


class ReportGenerator:
    """Generates comprehensive test reports in multiple formats"""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self, suite_result: TestSuiteResult, formats: List[str] = None
    ) -> Dict[str, str]:
        """
        Generate reports in specified formats

        Args:
            suite_result: Test suite results to report on
            formats: List of formats ('html', 'json', 'markdown', 'pdf')

        Returns:
            Dict mapping format names to file paths
        """
        if formats is None:
            formats = ["html", "json", "markdown"]

        generated_files = {}

        for fmt in formats:
            try:
                if fmt == "html":
                    file_path = self._generate_html_report(suite_result)
                elif fmt == "json":
                    file_path = self._generate_json_report(suite_result)
                elif fmt == "markdown":
                    file_path = self._generate_markdown_report(suite_result)
                elif fmt == "pdf":
                    file_path = self._generate_pdf_report(suite_result)
                else:
                    continue

                generated_files[fmt] = file_path

            except Exception as e:
                print(f"Failed to generate {fmt} report: {e}")

        return generated_files

    def _generate_html_report(self, suite_result: TestSuiteResult) -> str:
        """Generate HTML report with charts and detailed analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tentacle_test_report_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        # Generate charts if matplotlib is available
        chart_data = ""
        if MATPLOTLIB_AVAILABLE:
            chart_data = self._generate_charts_html(suite_result)

        # Generate performance data if available
        performance_data = ""
        if PERFORMANCE_AVAILABLE and suite_result.performance_summary:
            performance_data = self._generate_performance_html(suite_result)

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OctoBot Tentacles Test Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #007bff;
            margin: 0;
            font-size: 2.5em;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .status-passed {{ border-left-color: #28a745; }}
        .status-failed {{ border-left-color: #dc3545; }}
        .status-error {{ border-left-color: #ffc107; }}
        .charts {{
            margin: 30px 0;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .test-results {{
            margin-top: 30px;
        }}
        .test-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .test-table th,
        .test-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .test-table th {{
            background: #007bff;
            color: white;
            font-weight: 600;
        }}
        .status-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status-passed {{ background: #d4edda; color: #155724; }}
        .status-failed {{ background: #f8d7da; color: #721c24; }}
        .status-error {{ background: #fff3cd; color: #856404; }}
        .status-skipped {{ background: #e2e3e5; color: #383d41; }}
        .recommendations {{
            background: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
            border-left: 4px solid #007bff;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 OctoBot Tentacles Test Report</h1>
            <p><strong>Suite:</strong> {suite_result.suite_name}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Execution Mode:</strong> {suite_result.execution_mode.value}</p>
        </div>

        <div class="summary">
            <div class="metric-card">
                <div class="metric-value">{suite_result.total_tests}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card status-passed">
                <div class="metric-value">{suite_result.passed_tests}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-card status-failed">
                <div class="metric-value">{suite_result.failed_tests}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card status-error">
                <div class="metric-value">{suite_result.error_tests}</div>
                <div class="metric-label">Errors</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{suite_result.success_rate:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{suite_result.total_duration:.2f}s</div>
                <div class="metric-label">Total Duration</div>
            </div>
        </div>

        {chart_data}

        {performance_data}

        <div class="test-results">
            <h2>Test Results</h2>
            <table class="test-table">
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Timestamp</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add test results rows
        for result in suite_result.test_results:
            status_class = f"status-{result.status}"
            error_cell = result.error_message if result.error_message else "-"
            html_content += f"""
                    <tr>
                        <td>{result.test_name}</td>
                        <td>{result.test_type}</td>
                        <td><span class="status-badge {status_class}">{result.status.upper()}</span></td>
                        <td>{result.duration:.3f}s</td>
                        <td>{result.timestamp.strftime("%H:%M:%S")}</td>
                        <td>{error_cell}</td>
                    </tr>
"""

        html_content += """
                </tbody>
            </table>
        </div>

        <div class="recommendations">
            <h3>Recommendations</h3>
            <ul>
"""

        for rec in suite_result.recommendations:
            html_content += f"                <li>{rec}</li>\n"

        if not suite_result.recommendations:
            html_content += (
                "                <li>No specific recommendations at this time.</li>\n"
            )

        html_content += """
            </ul>
        </div>

        <div class="footer">
            <p>Report generated by OctoBot Tentacles Agent</p>
        </div>
    </div>
</body>
</html>
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filepath

    def _generate_charts_html(self, suite_result: TestSuiteResult) -> str:
        """Generate HTML with embedded charts"""
        if not suite_result.test_results:
            return ""

        # Create a simple bar chart using inline SVG
        status_counts = {
            "passed": suite_result.passed_tests,
            "failed": suite_result.failed_tests,
            "error": suite_result.error_tests,
            "skipped": suite_result.skipped_tests,
        }

        # Calculate chart dimensions
        bar_width = 60
        bar_spacing = 20
        chart_height = 200
        chart_width = (bar_width + bar_spacing) * len(status_counts) + 100

        max_count = max(status_counts.values()) if status_counts else 1

        svg_content = f'<svg width="{chart_width}" height="{chart_height + 50}" xmlns="http://www.w3.org/2000/svg">'

        # Add title
        svg_content += f'<text x="{chart_width // 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">Test Results by Status</text>'

        # Add bars
        x_pos = 50
        colors = {
            "passed": "#28a745",
            "failed": "#dc3545",
            "error": "#ffc107",
            "skipped": "#6c757d",
        }

        for status, count in status_counts.items():
            bar_height = (count / max_count) * chart_height if max_count > 0 else 0
            y_pos = chart_height + 30 - bar_height

            svg_content += f'<rect x="{x_pos}" y="{y_pos}" width="{bar_width}" height="{bar_height}" fill="{colors[status]}" />'
            svg_content += f'<text x="{x_pos + bar_width // 2}" y="{y_pos - 5}" text-anchor="middle" font-size="12">{count}</text>'
            svg_content += f'<text x="{x_pos + bar_width // 2}" y="{chart_height + 45}" text-anchor="middle" font-size="10">{status.title()}</text>'

            x_pos += bar_width + bar_spacing

        svg_content += "</svg>"

        return f"""
        <div class="charts">
            <h2>Test Analytics</h2>
            <div class="chart-container">
                <h3>Test Status Distribution</h3>
                {svg_content}
            </div>
        </div>
        """

    def _generate_performance_html(self, suite_result: TestSuiteResult) -> str:
        """Generate HTML with performance benchmarking data"""
        perf_summary = suite_result.performance_summary
        if not perf_summary:
            return ""

        regressions = perf_summary.get("regressions", [])
        operations = perf_summary.get("operations_summary", [])

        html = """
        <div class="charts">
            <h2>🚀 Performance Benchmarking</h2>
            <div class="chart-container">
                <h3>Performance Summary</h3>
                <div class="summary">
        """

        html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{perf_summary.get("total_operations", 0)}</div>
                        <div class="metric-label">Operations Benchmarked</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{perf_summary.get("total_baselines", 0)}</div>
                        <div class="metric-label">Performance Baselines</div>
                    </div>
                    <div class="metric-card {"status-error" if regressions else "status-passed"}">
                        <div class="metric-value">{len(regressions)}</div>
                        <div class="metric-label">Performance Regressions</div>
                    </div>
                </div>
        """

        if operations:
            html += """
                <h4>Operation Performance</h4>
                <table class="test-table">
                    <thead>
                        <tr>
                            <th>Operation</th>
                            <th>Duration (s)</th>
                            <th>Memory (MB)</th>
                            <th>CPU (%)</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for op in operations:
                html += f"""
                        <tr>
                            <td>{op.get("operation", "Unknown")}</td>
                            <td>{op.get("duration", 0):.3f}</td>
                            <td>{op.get("memory_mb", 0) or "-":.1f}</td>
                            <td>{op.get("cpu_percent", 0) or "-":.1f}</td>
                        </tr>
                """

            html += """
                    </tbody>
                </table>
            """

        if regressions:
            html += """
                <h4>⚠️ Performance Regressions Detected</h4>
                <div class="regressions">
            """

            for reg in regressions:
                html += f"""
                    <div class="recommendations">
                        <h4>{reg.get("operation", "Unknown Operation")}</h4>
                        <ul>
                            <li><strong>Duration Regression:</strong> {reg.get("duration_regression", 0):+.1f}%</li>
                """

                if reg.get("memory_regression"):
                    html += f"<li><strong>Memory Regression:</strong> {reg['memory_regression']:+.1f}%</li>"

                if reg.get("cpu_regression"):
                    html += f"<li><strong>CPU Regression:</strong> {reg['cpu_regression']:+.1f}%</li>"

                for rec in reg.get("recommendations", []):
                    html += f"<li>{rec}</li>"

                html += """
                        </ul>
                    </div>
                """

            html += """
                </div>
            """

        html += """
            </div>
        </div>
        """

        return html

    def _generate_json_report(self, suite_result: TestSuiteResult) -> str:
        """Generate JSON report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tentacle_test_report_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Convert dataclasses to dicts for JSON serialization
        report_data = {
            "suite_info": {
                "suite_id": suite_result.suite_id,
                "suite_name": suite_result.suite_name,
                "execution_mode": suite_result.execution_mode.value,
                "start_time": suite_result.start_time.isoformat(),
                "end_time": suite_result.end_time.isoformat()
                if suite_result.end_time
                else None,
                "total_duration": suite_result.total_duration,
            },
            "summary": {
                "total_tests": suite_result.total_tests,
                "passed_tests": suite_result.passed_tests,
                "failed_tests": suite_result.failed_tests,
                "error_tests": suite_result.error_tests,
                "skipped_tests": suite_result.skipped_tests,
                "success_rate": suite_result.success_rate,
                "average_test_duration": suite_result.average_test_duration,
            },
            "resource_usage": suite_result.resource_usage,
            "recommendations": suite_result.recommendations,
            "performance_summary": suite_result.performance_summary,
            "test_results": [
                {
                    "test_id": result.test_id,
                    "test_name": result.test_name,
                    "test_type": result.test_type,
                    "status": result.status,
                    "duration": result.duration,
                    "timestamp": result.timestamp.isoformat(),
                    "error_message": result.error_message,
                    "metrics": result.metrics,
                    "metadata": result.metadata,
                }
                for result in suite_result.test_results
            ],
            "generated_at": datetime.now().isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return filepath

    def _generate_markdown_report(self, suite_result: TestSuiteResult) -> str:
        """Generate Markdown report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tentacle_test_report_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)

        content = f"""# OctoBot Tentacles Test Report

**Suite:** {suite_result.suite_name}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Execution Mode:** {suite_result.execution_mode.value}  
**Duration:** {suite_result.total_duration:.2f}s  

## Summary

- **Total Tests:** {suite_result.total_tests}
- **Passed:** {suite_result.passed_tests}
- **Failed:** {suite_result.failed_tests}
- **Errors:** {suite_result.error_tests}
- **Skipped:** {suite_result.skipped_tests}
- **Success Rate:** {suite_result.success_rate:.1f}%
- **Average Duration:** {suite_result.average_test_duration:.3f}s

## Test Results

| Test Name | Type | Status | Duration | Error |
|-----------|------|--------|----------|-------|
"""

        for result in suite_result.test_results:
            error = result.error_message if result.error_message else "-"
            content += f"| {result.test_name} | {result.test_type} | {result.status} | {result.duration:.3f}s | {error} |\n"

        if suite_result.recommendations:
            content += "\n## Recommendations\n\n"
            for rec in suite_result.recommendations:
                content += f"- {rec}\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath

    def _generate_pdf_report(self, suite_result: TestSuiteResult) -> str:
        """Generate PDF report (placeholder - would require reportlab or similar)"""
        # For now, just create a text file indicating PDF generation would go here
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tentacle_test_report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        # Placeholder - in a real implementation, use reportlab or similar
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(
                "PDF Report Generation - Requires additional dependencies (reportlab, etc.)\n"
            )
            f.write(f"Suite: {suite_result.suite_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(
                f"Tests: {suite_result.total_tests}, Passed: {suite_result.passed_tests}\n"
            )

        return filepath


class CIDCIntegration:
    """CI/CD integration hooks and utilities"""

    def __init__(self, report_generator: ReportGenerator):
        self.report_generator = report_generator

    def generate_junit_xml(self, suite_result: TestSuiteResult) -> str:
        """Generate JUnit XML for CI/CD systems"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"junit_report_{timestamp}.xml"
        filepath = os.path.join(self.report_generator.output_dir, filename)

        # Calculate total time
        total_time = suite_result.total_duration

        xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="{suite_result.suite_name}"
               tests="{suite_result.total_tests}"
               failures="{suite_result.failed_tests}"
               errors="{suite_result.error_tests}"
               skipped="{suite_result.skipped_tests}"
               time="{total_time:.3f}"
               timestamp="{suite_result.start_time.isoformat()}">
'''

        for result in suite_result.test_results:
            status = result.status
            if status == "passed":
                xml_content += f'''        <testcase name="{result.test_name}"
                  classname="{result.test_type}"
                  time="{result.duration:.3f}">
        </testcase>
'''
            elif status == "failed":
                xml_content += f'''        <testcase name="{result.test_name}"
                  classname="{result.test_type}"
                  time="{result.duration:.3f}">
            <failure message="{result.error_message or "Test failed"}">
                {result.error_message or "Test failed"}
            </failure>
        </testcase>
'''
            elif status == "error":
                xml_content += f'''        <testcase name="{result.test_name}"
                  classname="{result.test_type}"
                  time="{result.duration:.3f}">
            <error message="{result.error_message or "Test error"}">
                {result.error_message or "Test error"}
            </error>
        </testcase>
'''
            elif status == "skipped":
                xml_content += f'''        <testcase name="{result.test_name}"
                  classname="{result.test_type}"
                  time="{result.duration:.3f}">
            <skipped message="Test skipped" />
        </testcase>
'''

        xml_content += """    </testsuite>
</testsuites>
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(xml_content)

        return filepath

    def check_quality_gates(self, suite_result: TestSuiteResult) -> Dict[str, Any]:
        """Check quality gates for CI/CD pipelines"""
        gates = {
            "success_rate_threshold": 90.0,  # Minimum 90% success rate
            "max_failures_allowed": 5,  # Maximum 5 failures allowed
            "max_errors_allowed": 2,  # Maximum 2 errors allowed
            "max_duration_seconds": 300,  # Maximum 5 minutes total
        }

        results = {
            "overall_pass": True,
            "gate_results": {},
            "blocking_issues": [],
        }

        # Check success rate
        success_rate_ok = suite_result.success_rate >= gates["success_rate_threshold"]
        results["gate_results"]["success_rate"] = {
            "required": gates["success_rate_threshold"],
            "actual": suite_result.success_rate,
            "passed": success_rate_ok,
        }
        if not success_rate_ok:
            results["overall_pass"] = False
            results["blocking_issues"].append(
                f"Success rate {suite_result.success_rate:.1f}% below threshold {gates['success_rate_threshold']}%"
            )

        # Check failure count
        failures_ok = suite_result.failed_tests <= gates["max_failures_allowed"]
        results["gate_results"]["max_failures"] = {
            "required": f"≤{gates['max_failures_allowed']}",
            "actual": suite_result.failed_tests,
            "passed": failures_ok,
        }
        if not failures_ok:
            results["overall_pass"] = False
            results["blocking_issues"].append(
                f"Too many failures: {suite_result.failed_tests} > {gates['max_failures_allowed']}"
            )

        # Check error count
        errors_ok = suite_result.error_tests <= gates["max_errors_allowed"]
        results["gate_results"]["max_errors"] = {
            "required": f"≤{gates['max_errors_allowed']}",
            "actual": suite_result.error_tests,
            "passed": errors_ok,
        }
        if not errors_ok:
            results["overall_pass"] = False
            results["blocking_issues"].append(
                f"Too many errors: {suite_result.error_tests} > {gates['max_errors_allowed']}"
            )

        # Check duration
        duration_ok = suite_result.total_duration <= gates["max_duration_seconds"]
        results["gate_results"]["max_duration"] = {
            "required": f"≤{gates['max_duration_seconds']}s",
            "actual": suite_result.total_duration,
            "passed": duration_ok,
        }
        if not duration_ok:
            results["overall_pass"] = False
            results["blocking_issues"].append(
                f"Test duration {suite_result.total_duration:.1f}s exceeds limit {gates['max_duration_seconds']}s"
            )

        return results

    def generate_github_summary(self, suite_result: TestSuiteResult) -> str:
        """Generate GitHub Actions job summary"""
        summary = f"""## 🧪 OctoBot Tentacles Test Results

**Suite:** {suite_result.suite_name}  
**Execution Mode:** {suite_result.execution_mode.value}  
**Duration:** {suite_result.total_duration:.2f}s  

### 📊 Summary
- ✅ **Passed:** {suite_result.passed_tests}
- ❌ **Failed:** {suite_result.failed_tests}
- ⚠️ **Errors:** {suite_result.error_tests}
- ⏭️ **Skipped:** {suite_result.skipped_tests}
- 📈 **Success Rate:** {suite_result.success_rate:.1f}%

"""

        if suite_result.failed_tests > 0 or suite_result.error_tests > 0:
            summary += "\n### 🚨 Issues\n"
            for result in suite_result.test_results:
                if result.status in ["failed", "error"]:
                    summary += f"- **{result.test_name}**: {result.error_message or 'Unknown error'}\n"

        if suite_result.recommendations:
            summary += "\n### 💡 Recommendations\n"
            for rec in suite_result.recommendations:
                summary += f"- {rec}\n"

        return summary


# Convenience functions
def create_test_suite(
    name: str,
    execution_mode: ExecutionMode = ExecutionMode.ASYNC,
    suite_id: Optional[str] = None,
) -> TestSuiteResult:
    """Create a new test suite"""
    if suite_id is None:
        suite_id = f"suite_{int(time.time())}"

    return TestSuiteResult(
        suite_id=suite_id,
        suite_name=name,
        execution_mode=execution_mode,
        start_time=datetime.now(),
    )


def generate_comprehensive_report(
    suite_result: TestSuiteResult,
    output_dir: str = "reports",
    formats: List[str] = None,
) -> Dict[str, str]:
    """Generate comprehensive reports in multiple formats"""
    generator = ReportGenerator(output_dir)
    return generator.generate_report(suite_result, formats)


def setup_ci_cd_integration(output_dir: str = "reports") -> CIDCIntegration:
    """Setup CI/CD integration"""
    generator = ReportGenerator(output_dir)
    return CIDCIntegration(generator)
