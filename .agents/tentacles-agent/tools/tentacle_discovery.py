#!/usr/bin/env python3
"""
Tentacle Discovery Utility

This utility provides dynamic discovery and import capabilities for OctoBot tentacles.
It scans the tentacles directory structure and provides functions to discover available
tentacle packages and their main classes.

Features:
- Dynamic scanning of tentacle packages
- Class extraction from packages
- Support for multiple classes per package
- Consumer/Producer subclass handling for trading modes
- Import path generation
- Metadata extraction without imports (fast discovery)
- Dependency graph analysis
- Caching system for performance
- Parallel discovery capabilities

Usage:
    from tentacle_discovery import TentacleDiscovery

    discovery = TentacleDiscovery()
    ta_evaluators = discovery.get_ta_evaluators()
    strategy_evaluators = discovery.get_strategy_evaluators()
    trading_modes = discovery.get_trading_modes()
    services = discovery.get_services()
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Any, Type, Optional, Tuple, Union
from pathlib import Path
import ast
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


class TentacleDiscovery:
    """Dynamic tentacle discovery and import utility"""

    def __init__(
        self, tentacles_path: Optional[Union[str, Path]] = None, use_cache: bool = True
    ):
        """
        Initialize tentacle discovery

        Args:
            tentacles_path: Path to OctoBot-Tentacles directory. If None, auto-detect.
            use_cache: Whether to use caching for performance
        """
        if tentacles_path is None:
            # Auto-detect tentacles path relative to this script
            script_dir = Path(__file__).parent
            tentacles_path = script_dir / "../../../OctoBot-Tentacles"

        self.tentacles_path = Path(tentacles_path).resolve()
        self.use_cache = use_cache
        self.cache_file = self.tentacles_path.parent / ".tentacle_discovery_cache.json"
        self.cache_ttl = 3600  # 1 hour cache TTL
        self._add_to_path()

    def _add_to_path(self):
        """Add tentacles and dependency paths to Python path for imports"""
        script_dir = Path(__file__).parent

        # Add all necessary paths like the existing tools do
        paths_to_add = [
            str(self.tentacles_path),  # OctoBot-Tentacles
            str(script_dir / "../../../OctoBot-Evaluators"),  # OctoBot-Evaluators
            str(script_dir / "../../../OctoBot-Commons"),  # OctoBot-Commons
            str(script_dir / "../../../Async-Channel"),  # Async-Channel
        ]

        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

    def _get_cache_key(self, tentacle_type: str) -> str:
        """Generate cache key for a tentacle type"""
        # Include file modification times in cache key
        base_path = self._get_base_path_for_type(tentacle_type)
        full_path = self.tentacles_path / base_path
        if not full_path.exists():
            return ""

        mtimes = []
        for item in full_path.rglob("*.py"):
            mtimes.append(str(item.stat().st_mtime))

        content_hash = hashlib.md5("".join(sorted(mtimes)).encode()).hexdigest()
        return f"{tentacle_type}_{content_hash}"

    def _load_cache(self) -> Dict[str, Any]:
        """Load cached discovery results"""
        if not self.use_cache or not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)

            # Check TTL
            if time.time() - cache_data.get("timestamp", 0) > self.cache_ttl:
                return {}

            return cache_data.get("data", {})
        except (json.JSONDecodeError, KeyError):
            return {}

    def _save_cache(self, data: Dict[str, Any]):
        """Save discovery results to cache"""
        if not self.use_cache:
            return

        cache_data = {"timestamp": time.time(), "data": data}

        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass  # Silently fail on cache write errors

    def _get_base_path_for_type(self, tentacle_type: str) -> str:
        """Get the base directory path for a tentacle type"""
        type_paths = {
            "TA": "Evaluator/TA",
            "STRATEGY": "Evaluator/Strategies",
            "TRADING_MODE": "Trading/Mode",
            "SERVICE": "Services/Services_bases",
        }
        return type_paths.get(tentacle_type, tentacle_type.lower())

    def _extract_metadata_from_package(
        self, package_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Extract class metadata from package without importing

        Args:
            package_path: Path to the package directory

        Returns:
            List of class metadata dicts with 'name', 'bases', 'docstring', etc.
        """
        classes = []

        # Check all Python files in the package
        for py_file in package_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue  # Skip __init__.py as it usually only has imports

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source, filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Extract base classes
                        bases = []
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                bases.append(base.id)
                            elif isinstance(base, ast.Attribute):
                                bases.append(
                                    f"{base.value.id}.{base.attr}"
                                    if isinstance(base.value, ast.Name)
                                    else base.attr
                                )

                        # Extract docstring
                        docstring = None
                        if (
                            node.body
                            and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Str)
                        ):
                            docstring = node.body[0].value.s

                        classes.append(
                            {
                                "name": node.name,
                                "bases": bases,
                                "docstring": docstring,
                                "lineno": node.lineno,
                                "file": str(py_file.relative_to(package_path)),
                                "has_consumer": node.name.endswith("Consumer"),
                                "has_producer": node.name.endswith("Producer"),
                            }
                        )

            except (SyntaxError, UnicodeDecodeError):
                continue

        return classes

    def _extract_dependencies_from_file(self, file_path: Path) -> List[str]:
        """
        Extract import dependencies from a Python file

        Args:
            file_path: Path to the Python file

        Returns:
            List of imported module names
        """
        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))
            dependencies = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module.split(".")[0])

            return list(set(dependencies))  # Remove duplicates

        except (SyntaxError, UnicodeDecodeError):
            return []

    def _build_dependency_graph(
        self, packages: List[str], tentacle_type: str
    ) -> Dict[str, List[str]]:
        """
        Build a dependency graph for tentacle packages

        Args:
            packages: List of package names
            tentacle_type: Type of tentacles

        Returns:
            Dict mapping package names to their dependencies
        """
        base_path = self._get_base_path_for_type(tentacle_type)
        graph = {}

        for package in packages:
            package_path = self.tentacles_path / base_path / package
            dependencies = []

            # Check all Python files in the package
            for py_file in package_path.rglob("*.py"):
                deps = self._extract_dependencies_from_file(py_file)
                # Filter for OctoBot-related dependencies
                octobot_deps = [
                    d
                    for d in deps
                    if d.startswith(("octobot", "tentacles", "async_channel"))
                ]
                dependencies.extend(octobot_deps)

            graph[package] = list(set(dependencies))  # Remove duplicates

        return graph

    def _scan_packages_parallel(
        self, base_path: str, max_workers: int = 4
    ) -> List[str]:
        """
        Scan for tentacle packages in parallel

        Args:
            base_path: Relative path from tentacles root (e.g., "Evaluator/TA")
            max_workers: Maximum number of parallel workers

        Returns:
            List of package names
        """
        full_path = self.tentacles_path / base_path
        if not full_path.exists():
            return []

        packages = []
        items = list(full_path.iterdir())

        def check_package(item):
            if item.is_dir() and (item / "__init__.py").exists():
                return item.name
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(check_package, item) for item in items]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    packages.append(result)

        return sorted(packages)

    def _discover_with_metadata(
        self, tentacle_type: str, use_parallel: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover tentacles using metadata extraction (fast path)

        Args:
            tentacle_type: Type of tentacles to discover
            use_parallel: Whether to use parallel processing

        Returns:
            Dict mapping package names to lists of class metadata
        """
        cache_key = f"metadata_{tentacle_type}"
        cache = self._load_cache()

        if cache_key in cache:
            return cache[cache_key]

        base_path = self._get_base_path_for_type(tentacle_type)
        scan_method = (
            self._scan_packages_parallel if use_parallel else self._scan_packages
        )
        packages = scan_method(base_path)

        result = {}

        def process_package(package):
            package_path = self.tentacles_path / base_path / package
            classes = self._extract_metadata_from_package(package_path)

            if classes:
                result[package] = []
                for cls_meta in classes:
                    # Filter out Consumer/Producer subclasses for trading modes
                    if tentacle_type == "TRADING_MODE":
                        if not (cls_meta["has_consumer"] or cls_meta["has_producer"]):
                            result[package].append(cls_meta)
                    else:
                        result[package].append(cls_meta)

        if use_parallel and len(packages) > 2:
            with ThreadPoolExecutor(max_workers=min(4, len(packages))) as executor:
                futures = [
                    executor.submit(process_package, package) for package in packages
                ]
                for future in as_completed(futures):
                    future.result()
        else:
            for package in packages:
                process_package(package)

        # Cache the results
        cache[cache_key] = result
        self._save_cache(cache)

        return result

    def _extract_classes_from_package(
        self, module_path: str, base_class: Optional[Type] = None
    ) -> List[Type]:
        """
        Extract classes from a tentacle package

        Args:
            module_path: Full module path (e.g., "tentacles.Evaluator.TA.momentum_evaluator")
            base_class: Optional base class to filter by inheritance

        Returns:
            List of classes found in the package
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"⚠️  Could not import {module_path}: {e}")
            return []

        classes = []
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and not name.startswith("_"):
                # Check if class is from this package or its submodules
                obj_module = getattr(obj, "__module__", "")
                if obj_module.startswith(module_path):
                    if base_class is None or issubclass(obj, base_class):
                        classes.append(obj)

        return classes

    def _get_import_path(
        self, package_name: str, class_name: str, tentacle_type: str
    ) -> str:
        """
        Generate the full import path for a tentacle class

        Args:
            package_name: Name of the package (e.g., "momentum_evaluator")
            class_name: Name of the class (e.g., "RSIMomentumEvaluator")
            tentacle_type: Type category ("TA", "STRATEGY", "TRADING_MODE", "SERVICE")

        Returns:
            Full import path (e.g., "from tentacles.Evaluator.TA.momentum_evaluator import RSIMomentumEvaluator")
        """
        type_paths = {
            "TA": "Evaluator.TA",
            "STRATEGY": "Evaluator.Strategies",
            "TRADING_MODE": "Trading.Mode",
            "SERVICE": "Services.Services_bases",
        }

        base_path = type_paths.get(tentacle_type, tentacle_type.lower())
        return f"from tentacles.{base_path}.{package_name} import {class_name}"

    def get_ta_evaluators(
        self, use_metadata: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available TA evaluators

        Args:
            use_metadata: If True, use fast metadata extraction; if False, use full imports

        Returns:
            Dict mapping package names to lists of class info dicts
            Each dict contains: 'name', 'class' (if imported), 'import_path', 'bases', 'docstring', etc.
        """
        if use_metadata:
            # Try metadata-based discovery first
            metadata_result = self._discover_with_metadata("TA")
            if metadata_result:
                # Enhance with import paths and class objects where possible
                return self._enhance_metadata_with_imports(metadata_result, "TA")

        # Fallback to full import method
        return self._get_ta_evaluators_full_import()

    def _get_ta_evaluators_full_import(self) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy method using full imports"""
        packages = self._scan_packages("Evaluator/TA")
        result = {}

        for package in packages:
            module_path = f"tentacles.Evaluator.TA.{package}"
            classes = self._extract_classes_from_package(module_path)

            if classes:
                result[package] = []
                for cls in classes:
                    result[package].append(
                        {
                            "name": cls.__name__,
                            "class": cls,
                            "import_path": self._get_import_path(
                                package, cls.__name__, "TA"
                            ),
                        }
                    )

        return result

    def _enhance_metadata_with_imports(
        self, metadata_result: Dict[str, List[Dict[str, Any]]], tentacle_type: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enhance metadata with actual class objects and import paths where possible

        Args:
            metadata_result: Metadata from discovery
            tentacle_type: Type of tentacles

        Returns:
            Enhanced result with class objects where import succeeds
        """
        enhanced = {}

        for package, classes in metadata_result.items():
            enhanced[package] = []
            for cls_meta in classes:
                enhanced_cls = cls_meta.copy()
                enhanced_cls["import_path"] = self._get_import_path(
                    package, cls_meta["name"], tentacle_type
                )

                # Try to import the class object
                try:
                    module_path = f"tentacles.{self._get_base_path_for_type(tentacle_type).replace('/', '.')}.{package}"
                    module = importlib.import_module(module_path)
                    cls_obj = getattr(module, cls_meta["name"], None)
                    if cls_obj:
                        enhanced_cls["class"] = cls_obj
                except (ImportError, AttributeError):
                    pass  # Keep metadata-only if import fails

                enhanced[package].append(enhanced_cls)

        return enhanced

    def _scan_packages(self, base_path: str) -> List[str]:
        """
        Scan for tentacle packages in a directory

        Args:
            base_path: Relative path from tentacles root (e.g., "Evaluator/TA")

        Returns:
            List of package names
        """
        full_path = self.tentacles_path / base_path
        if not full_path.exists():
            return []

        packages = []
        for item in full_path.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                packages.append(item.name)

        return sorted(packages)

    def get_strategy_evaluators(
        self, use_metadata: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available strategy evaluators

        Args:
            use_metadata: If True, use fast metadata extraction; if False, use full imports

        Returns:
            Dict mapping package names to lists of class info dicts
        """
        if use_metadata:
            metadata_result = self._discover_with_metadata("STRATEGY")
            if metadata_result:
                return self._enhance_metadata_with_imports(metadata_result, "STRATEGY")

        return self._get_strategy_evaluators_full_import()

    def _get_strategy_evaluators_full_import(self) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy method using full imports"""
        packages = self._scan_packages("Evaluator/Strategies")
        result = {}

        for package in packages:
            module_path = f"tentacles.Evaluator.Strategies.{package}"
            classes = self._extract_classes_from_package(module_path)

            if classes:
                result[package] = []
                for cls in classes:
                    result[package].append(
                        {
                            "name": cls.__name__,
                            "class": cls,
                            "import_path": self._get_import_path(
                                package, cls.__name__, "STRATEGY"
                            ),
                        }
                    )

        return result

    def get_trading_modes(
        self, use_metadata: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available trading modes

        Args:
            use_metadata: If True, use fast metadata extraction; if False, use full imports

        Returns:
            Dict mapping package names to lists of class info dicts
            Handles Consumer/Producer subclasses appropriately
        """
        if use_metadata:
            metadata_result = self._discover_with_metadata("TRADING_MODE")
            if metadata_result:
                return self._enhance_metadata_with_imports(
                    metadata_result, "TRADING_MODE"
                )

        return self._get_trading_modes_full_import()

    def _get_trading_modes_full_import(self) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy method using full imports"""
        packages = self._scan_packages("Trading/Mode")
        result = {}

        for package in packages:
            module_path = f"tentacles.Trading.Mode.{package}"
            classes = self._extract_classes_from_package(module_path)

            if classes:
                result[package] = []
                for cls in classes:
                    # Filter out Consumer/Producer subclasses, keep main mode classes
                    if not cls.__name__.endswith(
                        "Consumer"
                    ) and not cls.__name__.endswith("Producer"):
                        result[package].append(
                            {
                                "name": cls.__name__,
                                "class": cls,
                                "import_path": self._get_import_path(
                                    package, cls.__name__, "TRADING_MODE"
                                ),
                                "has_consumer": any(
                                    c.__name__.endswith("Consumer") for c in classes
                                ),
                                "has_producer": any(
                                    c.__name__.endswith("Producer") for c in classes
                                ),
                            }
                        )

        return result

    def get_services(
        self, use_metadata: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available services

        Args:
            use_metadata: If True, use fast metadata extraction; if False, use full imports

        Returns:
            Dict mapping package names to lists of class info dicts
        """
        if use_metadata:
            metadata_result = self._discover_with_metadata("SERVICE")
            if metadata_result:
                return self._enhance_metadata_with_imports(metadata_result, "SERVICE")

        return self._get_services_full_import()

    def _get_services_full_import(self) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy method using full imports"""
        packages = self._scan_packages("Services/Services_bases")
        result = {}

        for package in packages:
            module_path = f"tentacles.Services.Services_bases.{package}"
            classes = self._extract_classes_from_package(module_path)

            if classes:
                result[package] = []
                for cls in classes:
                    result[package].append(
                        {
                            "name": cls.__name__,
                            "class": cls,
                            "import_path": self._get_import_path(
                                package, cls.__name__, "SERVICE"
                            ),
                        }
                    )

        return result

    def get_dependency_graph(self, tentacle_type: str) -> Dict[str, List[str]]:
        """
        Get dependency graph for a tentacle type

        Args:
            tentacle_type: Type of tentacles ("TA", "STRATEGY", "TRADING_MODE", "SERVICE")

        Returns:
            Dict mapping package names to their dependencies
        """
        cache_key = f"deps_{tentacle_type}"
        cache = self._load_cache()

        if cache_key in cache:
            return cache[cache_key]

        packages = self._scan_packages(self._get_base_path_for_type(tentacle_type))
        graph = self._build_dependency_graph(packages, tentacle_type)

        # Cache the result
        cache[cache_key] = graph
        self._save_cache(cache)

        return graph

    def clear_cache(self):
        """Clear the discovery cache"""
        if self.cache_file.exists():
            self.cache_file.unlink()

    def get_discovery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the discovery process

        Returns:
            Dict with discovery statistics
        """
        stats = {
            "cache_enabled": self.use_cache,
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists(),
            "cache_size": self.cache_file.stat().st_size
            if self.cache_file.exists()
            else 0,
        }

        if self.cache_file.exists():
            try:
                cache_data = self._load_cache()
                stats["cache_timestamp"] = cache_data.get("timestamp", 0)
                stats["cached_types"] = list(cache_data.get("data", {}).keys())
            except:
                stats["cache_valid"] = False
        else:
            stats["cache_valid"] = True

        return stats

    def find_class_by_name(
        self, class_name: str, tentacle_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a specific class by name across all tentacle types

        Args:
            class_name: Name of the class to find
            tentacle_type: Optional type filter ("TA", "STRATEGY", "TRADING_MODE", "SERVICE")

        Returns:
            Class info dict or None if not found
        """
        search_funcs = []
        if tentacle_type is None:
            search_funcs = [
                self.get_ta_evaluators,
                self.get_strategy_evaluators,
                self.get_trading_modes,
                self.get_services,
            ]
        else:
            type_map = {
                "TA": self.get_ta_evaluators,
                "STRATEGY": self.get_strategy_evaluators,
                "TRADING_MODE": self.get_trading_modes,
                "SERVICE": self.get_services,
            }
            search_funcs = [type_map.get(tentacle_type, lambda: {})]

        for func in search_funcs:
            data = func()
            for package, classes in data.items():
                for cls_info in classes:
                    if cls_info["name"] == class_name:
                        return cls_info

        return None

    def print_discovery_summary(self):
        """Print a summary of discovered tentacles"""
        print("🔍 Tentacle Discovery Summary")
        print("=" * 40)

        ta = self.get_ta_evaluators()
        strategy = self.get_strategy_evaluators()
        trading = self.get_trading_modes()
        services = self.get_services()

        print(
            f"📊 TA Evaluators: {sum(len(classes) for classes in ta.values())} classes in {len(ta)} packages"
        )
        for package, classes in ta.items():
            print(f"   {package}: {len(classes)} classes")

        print(
            f"🧠 Strategy Evaluators: {sum(len(classes) for classes in strategy.values())} classes in {len(strategy)} packages"
        )
        for package, classes in strategy.items():
            print(f"   {package}: {len(classes)} classes")

        print(
            f"📈 Trading Modes: {sum(len(classes) for classes in trading.values())} classes in {len(trading)} packages"
        )
        for package, classes in trading.items():
            print(f"   {package}: {len(classes)} classes")

        print(
            f"🔧 Services: {sum(len(classes) for classes in services.values())} classes in {len(services)} packages"
        )
        for package, classes in services.items():
            print(f"   {package}: {len(classes)} classes")

        print("=" * 40)


# Utility functions for easy access
def get_all_ta_evaluators() -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to get all TA evaluators"""
    discovery = TentacleDiscovery()
    return discovery.get_ta_evaluators()


def get_all_strategy_evaluators() -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to get all strategy evaluators"""
    discovery = TentacleDiscovery()
    return discovery.get_strategy_evaluators()


def get_all_trading_modes() -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to get all trading modes"""
    discovery = TentacleDiscovery()
    return discovery.get_trading_modes()


def get_all_services() -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to get all services"""
    discovery = TentacleDiscovery()
    return discovery.get_services()


def find_tentacle_class(
    class_name: str, tentacle_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Convenience function to find a specific tentacle class"""
    discovery = TentacleDiscovery()
    return discovery.find_class_by_name(class_name, tentacle_type)


if __name__ == "__main__":
    # Demo the discovery functionality
    discovery = TentacleDiscovery()
    discovery.print_discovery_summary()

    print("\n🔍 Example TA Evaluator Classes:")
    ta_evaluators = discovery.get_ta_evaluators()
    for package, classes in list(ta_evaluators.items())[:2]:  # Show first 2 packages
        print(f"  {package}:")
        for cls_info in classes[:3]:  # Show first 3 classes per package
            print(f"    - {cls_info['name']}")
            print(f"      Import: {cls_info['import_path']}")
