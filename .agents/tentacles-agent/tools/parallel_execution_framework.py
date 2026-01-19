"""
Parallel Execution Framework for OctoBot Tentacles Agent

This module provides scalable parallel execution capabilities for tentacle testing,
including concurrent processing, resource management, and distributed task execution.
"""

import asyncio
import concurrent.futures
import threading
import time
import psutil
import os
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for parallel processing"""

    SEQUENTIAL = "sequential"
    ASYNC = "async"
    MULTIPROCESS = "multiprocess"
    HYBRID = "hybrid"  # Async + Multiprocess


class TaskStatus(Enum):
    """Status of individual tasks"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a single test task"""

    task_id: str
    name: str
    func: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0

    def __post_init__(self):
        pass

    @property
    def duration(self) -> Optional[float]:
        """Get task execution duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def is_finished(self) -> bool:
        """Check if task is finished (completed, failed, or cancelled)"""
        return self.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ]


class ResourceManager:
    """Manages system resources during parallel execution"""

    def __init__(
        self,
        max_memory_percent: float = 80.0,
        max_cpu_percent: float = 90.0,
        max_concurrent_tasks: Optional[int] = None,
        enable_monitoring: bool = True,
    ):
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        self.max_concurrent_tasks = max_concurrent_tasks or (os.cpu_count() or 4) * 2
        self.enable_monitoring = enable_monitoring

        # Resource tracking
        self.memory_usage_history = []
        self.cpu_usage_history = []
        self.active_tasks = 0

        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False

    def start_monitoring(self):
        """Start resource monitoring thread"""
        if self.enable_monitoring and not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitor_resources, daemon=True
            )
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

    def _monitor_resources(self):
        """Monitor system resources in background thread"""
        while self.monitoring_active:
            try:
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)

                self.memory_usage_history.append((time.time(), memory_percent))
                self.cpu_usage_history.append((time.time(), cpu_percent))

                # Keep only last 100 readings
                if len(self.memory_usage_history) > 100:
                    self.memory_usage_history = self.memory_usage_history[-100:]
                if len(self.cpu_usage_history) > 100:
                    self.cpu_usage_history = self.cpu_usage_history[-100:]

                # Check resource limits
                if memory_percent > self.max_memory_percent:
                    logger.warning(".1f")
                if cpu_percent > self.max_cpu_percent:
                    logger.warning(".1f")

                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(10)

    def can_start_task(self) -> bool:
        """Check if a new task can be started based on resource limits"""
        if self.active_tasks >= self.max_concurrent_tasks:
            return False

        if self.enable_monitoring:
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.1)

            if memory_percent > self.max_memory_percent:
                return False
            if cpu_percent > self.max_cpu_percent:
                return False

        return True

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        return {
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "active_tasks": self.active_tasks,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "memory_limit": self.max_memory_percent,
            "cpu_limit": self.max_cpu_percent,
        }


class ParallelExecutor:
    """Main parallel execution engine for tentacle testing"""

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.ASYNC,
        max_workers: Optional[int] = None,
        resource_manager: Optional[ResourceManager] = None,
    ):
        self.mode = mode
        self.max_workers = max_workers or (os.cpu_count() or 4)
        self.resource_manager = resource_manager or ResourceManager()

        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.completed_tasks: Dict[str, Task] = {}

        # Execution state
        self.is_running = False
        self.executor = None
        self.event_loop = None

        # Progress tracking
        self.progress_callbacks: List[Callable] = []

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def start(self):
        """Start the parallel executor"""
        if self.is_running:
            return

        self.is_running = True
        self.resource_manager.start_monitoring()

        if self.mode == ExecutionMode.MULTIPROCESS:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )
        elif self.mode == ExecutionMode.HYBRID:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            )

        logger.info(
            f"Started parallel executor in {self.mode.value} mode with {self.max_workers} workers"
        )

    async def stop(self):
        """Stop the parallel executor"""
        if not self.is_running:
            return

        self.is_running = False
        self.resource_manager.stop_monitoring()

        if self.executor:
            self.executor.shutdown(wait=True)

        logger.info("Stopped parallel executor")

    def add_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        *args,
        priority: int = 1,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Add a task to the execution queue"""
        task = Task(
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
        )

        self.tasks[task_id] = task
        self.task_queue.put_nowait((priority, task_id, task))

        logger.debug(f"Added task {task_id}: {name}")
        return task_id

    def add_progress_callback(self, callback: Callable):
        """Add a callback for progress updates"""
        self.progress_callbacks.append(callback)

    def _notify_progress(self, task: Task):
        """Notify progress callbacks of task updates"""
        for callback in self.progress_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    async def execute_all(self) -> Dict[str, Any]:
        """Execute all queued tasks and return results"""
        if not self.is_running:
            await self.start()

        results = {}

        if self.mode == ExecutionMode.SEQUENTIAL:
            results = await self._execute_sequential()
        elif self.mode == ExecutionMode.ASYNC:
            results = await self._execute_async()
        elif self.mode == ExecutionMode.MULTIPROCESS:
            results = await self._execute_multiprocess()
        elif self.mode == ExecutionMode.HYBRID:
            results = await self._execute_hybrid()

        return results

    async def _execute_sequential(self) -> Dict[str, Any]:
        """Execute tasks sequentially"""
        results = {}

        while not self.task_queue.empty():
            try:
                _, task_id, task = self.task_queue.get_nowait()
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()
                self._notify_progress(task)

                try:
                    if asyncio.iscoroutinefunction(task.func):
                        if task.timeout:
                            task.result = await asyncio.wait_for(
                                task.func(*task.args, **task.kwargs),
                                timeout=task.timeout,
                            )
                        else:
                            task.result = await task.func(*task.args, **task.kwargs)
                    else:
                        if task.timeout:
                            task.result = await asyncio.wait_for(
                                asyncio.get_event_loop().run_in_executor(
                                    None, task.func, *task.args, **task.kwargs
                                ),
                                timeout=task.timeout,
                            )
                        else:
                            task.result = (
                                await asyncio.get_event_loop().run_in_executor(
                                    None, task.func, *task.args, **task.kwargs
                                )
                            )

                    task.status = TaskStatus.COMPLETED
                    results[task_id] = {"success": True, "result": task.result}

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = e
                    results[task_id] = {"success": False, "error": str(e)}

                finally:
                    task.end_time = time.time()
                    task.progress = 1.0
                    self._notify_progress(task)
                    self.completed_tasks[task_id] = task

            except asyncio.QueueEmpty:
                break

        return results

    async def _execute_async(self) -> Dict[str, Any]:
        """Execute tasks asynchronously"""

        async def run_task(task: Task) -> Dict[str, Any]:
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            self.resource_manager.active_tasks += 1
            self._notify_progress(task)

            try:
                if asyncio.iscoroutinefunction(task.func):
                    if task.timeout:
                        task.result = await asyncio.wait_for(
                            task.func(*task.args, **task.kwargs), timeout=task.timeout
                        )
                    else:
                        task.result = await task.func(*task.args, **task.kwargs)
                else:
                    if task.timeout:
                        task.result = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                self.executor, task.func, *task.args, **task.kwargs
                            ),
                            timeout=task.timeout,
                        )
                    else:
                        task.result = await asyncio.get_event_loop().run_in_executor(
                            self.executor, task.func, *task.args, **task.kwargs
                        )

                task.status = TaskStatus.COMPLETED
                return {"success": True, "result": task.result}

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = e
                return {"success": False, "error": str(e)}

            finally:
                task.end_time = time.time()
                task.progress = 1.0
                self.resource_manager.active_tasks -= 1
                self._notify_progress(task)
                self.completed_tasks[task.task_id] = task

        # Create semaphore for resource management
        semaphore = asyncio.Semaphore(self.resource_manager.max_concurrent_tasks)

        async def run_with_semaphore(task: Task) -> Dict[str, Any]:
            async with semaphore:
                return await run_task(task)

        # Execute all tasks with concurrency control
        pending_tasks = []
        while not self.task_queue.empty():
            try:
                _, task_id, task = self.task_queue.get_nowait()
                if self.resource_manager.can_start_task():
                    pending_tasks.append(run_with_semaphore(task))
                else:
                    # Wait a bit and try again
                    await asyncio.sleep(0.1)
                    self.task_queue.put_nowait((task.priority, task_id, task))
            except asyncio.QueueEmpty:
                break

        # Wait for all tasks to complete
        results = {}
        if pending_tasks:
            task_results = await asyncio.gather(*pending_tasks, return_exceptions=True)
            for i, result in enumerate(task_results):
                task_id = list(self.tasks.keys())[i]
                if isinstance(result, Exception):
                    results[task_id] = {"success": False, "error": str(result)}
                else:
                    results[task_id] = result

        return results

    async def _execute_multiprocess(self) -> Dict[str, Any]:
        """Execute tasks using multiprocessing"""
        if not self.executor:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )

        # For multiprocessing, we need to be careful with async functions
        # Convert async functions to sync wrappers
        def sync_wrapper(func, *args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                # This is a limitation - async functions can't easily be run in processes
                # We'll need to convert them or fall back to threading
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(func(*args, **kwargs))
                finally:
                    loop.close()
            else:
                return func(*args, **kwargs)

        # Execute tasks
        future_to_task = {}
        results = {}

        while not self.task_queue.empty():
            try:
                _, task_id, task = self.task_queue.get_nowait()

                if self.resource_manager.can_start_task():
                    task.status = TaskStatus.RUNNING
                    task.start_time = time.time()
                    self.resource_manager.active_tasks += 1
                    self._notify_progress(task)

                    future = self.executor.submit(
                        sync_wrapper, task.func, *task.args, **task.kwargs
                    )
                    future_to_task[future] = task

            except asyncio.QueueEmpty:
                break

        # Wait for completion
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            task.end_time = time.time()
            self.resource_manager.active_tasks -= 1

            try:
                task.result = future.result(timeout=task.timeout)
                task.status = TaskStatus.COMPLETED
                results[task.task_id] = {"success": True, "result": task.result}
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = e
                results[task.task_id] = {"success": False, "error": str(e)}

            task.progress = 1.0
            self._notify_progress(task)
            self.completed_tasks[task.task_id] = task

        return results

    async def _execute_hybrid(self) -> Dict[str, Any]:
        """Execute tasks using hybrid async + multiprocessing approach"""
        # For hybrid mode, use async for I/O bound tasks and multiprocessing for CPU bound tasks
        # This is a simplified implementation - in practice, we'd need task classification
        return await self._execute_async()

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a specific task"""
        task = self.tasks.get(task_id)
        return task.status if task else None

    def get_all_tasks(self) -> Dict[str, Task]:
        """Get all tasks (active and completed)"""
        return {**self.tasks, **self.completed_tasks}

    def get_active_tasks(self) -> Dict[str, Task]:
        """Get currently active tasks"""
        return {
            tid: task
            for tid, task in self.tasks.items()
            if task.status == TaskStatus.RUNNING
        }

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        task = self.tasks.get(task_id)
        if task and not task.is_finished:
            task.status = TaskStatus.CANCELLED
            task.end_time = time.time()
            self.completed_tasks[task_id] = task
            logger.info(f"Cancelled task {task_id}")
            return True
        return False

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        all_tasks = self.get_all_tasks()
        completed_count = sum(
            1 for t in all_tasks.values() if t.status == TaskStatus.COMPLETED
        )
        failed_count = sum(
            1 for t in all_tasks.values() if t.status == TaskStatus.FAILED
        )
        cancelled_count = sum(
            1 for t in all_tasks.values() if t.status == TaskStatus.CANCELLED
        )

        completed_tasks = [
            t
            for t in all_tasks.values()
            if t.status == TaskStatus.COMPLETED and t.duration is not None
        ]

        total_duration = sum(
            t.duration for t in completed_tasks if t.duration is not None
        )

        return {
            "total_tasks": len(all_tasks),
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "cancelled_tasks": cancelled_count,
            "running_tasks": len(self.get_active_tasks()),
            "pending_tasks": sum(
                1 for t in all_tasks.values() if t.status == TaskStatus.PENDING
            ),
            "average_duration": total_duration / len(completed_tasks)
            if completed_tasks
            else 0,
            "resource_stats": self.resource_manager.get_resource_stats(),
        }


# Convenience functions for easy use
async def run_parallel_tasks(
    tasks: List[Dict[str, Any]],
    mode: ExecutionMode = ExecutionMode.ASYNC,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run multiple tasks in parallel

    Args:
        tasks: List of task dicts with 'id', 'name', 'func', 'args', 'kwargs' keys
        mode: Execution mode
        max_workers: Maximum number of workers

    Returns:
        Dict of task results
    """
    async with ParallelExecutor(mode=mode, max_workers=max_workers) as executor:
        for task_data in tasks:
            executor.add_task(
                task_id=task_data["id"],
                name=task_data["name"],
                func=task_data["func"],
                *task_data.get("args", []),
                **task_data.get("kwargs", {}),
            )

        return await executor.execute_all()


def create_tentacle_test_tasks(
    tentacle_combinations: List[Dict[str, Any]], test_type: str = "validation"
) -> List[Dict[str, Any]]:
    """
    Create parallel tasks for tentacle testing

    Args:
        tentacle_combinations: List of tentacle combinations to test
        test_type: Type of test ('validation', 'benchmark', 'stress')

    Returns:
        List of task configurations
    """
    tasks = []

    for i, combo in enumerate(tentacle_combinations):
        task = {
            "id": f"{test_type}_test_{i}",
            "name": f"Test combination {i + 1}: {len(combo)} tentacles",
            "func": lambda c=combo: f"Testing combination: {c}",  # Placeholder
            "args": [],
            "kwargs": {"combination": combo, "test_type": test_type},
        }
        tasks.append(task)

    return tasks
