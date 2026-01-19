"""
Web Interface for OctoBot Tentacles Agent

Provides a web-based dashboard for managing and monitoring tentacle testing,
with real-time progress updates and comprehensive result visualization.
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

try:
    from flask import Flask, render_template, request, jsonify, Response
    from flask_socketio import SocketIO, emit

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available - web interface disabled")

from parallel_execution_framework import (
    ParallelExecutor,
    ExecutionMode,
    ResourceManager,
    create_tentacle_test_tasks,
    run_parallel_tasks,
)


class TentacleTestSession:
    """Represents a test session with multiple parallel tasks"""

    def __init__(self, session_id: str, name: str, test_type: str = "validation"):
        self.session_id = session_id
        self.name = name
        self.test_type = test_type
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status = "created"  # created, running, completed, failed

        # Test configuration
        self.tentacle_combinations: List[Dict[str, Any]] = []
        self.execution_mode = ExecutionMode.ASYNC
        self.max_workers = None

        # Results
        self.results: Dict[str, Any] = {}
        self.progress = 0.0
        self.errors: List[str] = []

        # Executor
        self.executor: Optional[ParallelExecutor] = None
        self.executor_task: Optional[asyncio.Task] = None

    def add_tentacle_combination(self, combination: Dict[str, Any]):
        """Add a tentacle combination to test"""
        self.tentacle_combinations.append(combination)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "test_type": self.test_type,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "status": self.status,
            "tentacle_combinations": self.tentacle_combinations,
            "execution_mode": self.execution_mode.value,
            "max_workers": self.max_workers,
            "progress": self.progress,
            "results": self.results,
            "errors": self.errors,
        }


class TentacleTestManager:
    """Manages multiple test sessions and provides web interface integration"""

    def __init__(self):
        self.sessions: Dict[str, TentacleTestSession] = {}
        self.event_loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    def create_session(self, name: str, test_type: str = "validation") -> str:
        """Create a new test session"""
        session_id = str(uuid.uuid4())
        session = TentacleTestSession(session_id, name, test_type)
        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[TentacleTestSession]:
        """Get a session by ID"""
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        return [session.to_dict() for session in self.sessions.values()]

    async def start_session(
        self,
        session_id: str,
        execution_mode: ExecutionMode = ExecutionMode.ASYNC,
        max_workers: Optional[int] = None,
    ) -> bool:
        """Start a test session"""
        session = self.get_session(session_id)
        if not session or session.status != "created":
            return False

        session.execution_mode = execution_mode
        session.max_workers = max_workers
        session.started_at = datetime.now()
        session.status = "running"

        # Create tasks from tentacle combinations
        tasks = create_tentacle_test_tasks(
            session.tentacle_combinations, session.test_type
        )

        # Create executor
        resource_manager = ResourceManager()
        session.executor = ParallelExecutor(
            mode=execution_mode,
            max_workers=max_workers,
            resource_manager=resource_manager,
        )

        # Add progress callback
        def progress_callback(task):
            session.progress = sum(
                t.progress for t in session.executor.get_all_tasks().values()
            ) / len(session.executor.get_all_tasks())
            # Emit progress update via socketio if available
            if hasattr(self, "socketio"):
                self.socketio.emit(
                    "progress_update",
                    {
                        "session_id": session_id,
                        "progress": session.progress,
                        "task_status": task.status.value,
                    },
                )

        session.executor.add_progress_callback(progress_callback)

        # Start executor and run tasks
        await session.executor.start()

        # Add tasks to executor
        for task_data in tasks:
            session.executor.add_task(
                task_id=task_data["id"],
                name=task_data["name"],
                func=task_data["func"],
                *task_data.get("args", []),
                **task_data.get("kwargs", {}),
            )

        # Execute tasks
        session.executor_task = asyncio.create_task(session.executor.execute_all())

        def on_completion(task):
            try:
                session.results = task.result()
                session.completed_at = datetime.now()
                session.status = "completed"
                session.progress = 1.0
            except Exception as e:
                session.errors.append(str(e))
                session.status = "failed"
                session.completed_at = datetime.now()

        session.executor_task.add_done_callback(on_completion)

        return True

    async def stop_session(self, session_id: str) -> bool:
        """Stop a running session"""
        session = self.get_session(session_id)
        if not session or session.status != "running":
            return False

        if session.executor:
            await session.executor.stop()

        session.status = "stopped"
        session.completed_at = datetime.now()
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.status == "running" and session.executor:
                # Cancel the executor task
                if session.executor_task and not session.executor_task.done():
                    session.executor_task.cancel()
            del self.sessions[session_id]
            return True
        return False

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a session"""
        session = self.get_session(session_id)
        if not session:
            return None

        stats = session.to_dict()

        if session.executor:
            executor_stats = session.executor.get_execution_stats()
            stats.update(executor_stats)

        return stats


# Global test manager instance
test_manager = TentacleTestManager()


if FLASK_AVAILABLE:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    socketio = SocketIO(app, cors_allowed_origins="*")

    # Store socketio reference in manager
    test_manager.socketio = socketio

    @app.route("/")
    def index():
        """Main dashboard page"""
        return render_template("dashboard.html")

    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        """List all test sessions"""
        sessions = test_manager.list_sessions()
        return jsonify({"sessions": sessions})

    @app.route("/api/sessions", methods=["POST"])
    def create_session():
        """Create a new test session"""
        data = request.get_json()
        name = data.get(
            "name", f"Test Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        test_type = data.get("test_type", "validation")

        session_id = test_manager.create_session(name, test_type)
        return jsonify({"session_id": session_id, "status": "created"})

    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def get_session(session_id):
        """Get session details"""
        session = test_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        return jsonify(session.to_dict())

    @app.route("/api/sessions/<session_id>/start", methods=["POST"])
    def start_session(session_id):
        """Start a test session"""
        data = request.get_json()
        execution_mode = ExecutionMode(data.get("execution_mode", "async"))
        max_workers = data.get("max_workers")

        # Schedule the async start
        async def _start():
            return await test_manager.start_session(
                session_id, execution_mode, max_workers
            )

        # Run in the manager's event loop
        future = asyncio.run_coroutine_threadsafe(_start(), test_manager.event_loop)
        success = future.result(timeout=10)

        if success:
            return jsonify({"status": "started"})
        else:
            return jsonify({"error": "Failed to start session"}), 400

    @app.route("/api/sessions/<session_id>/stop", methods=["POST"])
    def stop_session(session_id):
        """Stop a test session"""

        async def _stop():
            return await test_manager.stop_session(session_id)

        future = asyncio.run_coroutine_threadsafe(_stop(), test_manager.event_loop)
        success = future.result(timeout=10)

        if success:
            return jsonify({"status": "stopped"})
        else:
            return jsonify({"error": "Failed to stop session"}), 400

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def delete_session(session_id):
        """Delete a test session"""
        success = test_manager.delete_session(session_id)
        if success:
            return jsonify({"status": "deleted"})
        else:
            return jsonify({"error": "Session not found"}), 404

    @app.route("/api/sessions/<session_id>/stats", methods=["GET"])
    def get_session_stats(session_id):
        """Get detailed session statistics"""
        stats = test_manager.get_session_stats(session_id)
        if stats:
            return jsonify(stats)
        else:
            return jsonify({"error": "Session not found"}), 404

    @app.route("/api/sessions/<session_id>/combinations", methods=["POST"])
    def add_combination(session_id):
        """Add tentacle combination to session"""
        session = test_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        combination = request.get_json()
        session.add_tentacle_combination(combination)

        return jsonify({"status": "added"})

    @socketio.on("connect")
    def handle_connect():
        """Handle client connection"""
        emit("connected", {"status": "connected"})

    @socketio.on("subscribe_session")
    def handle_subscribe_session(data):
        """Subscribe to session progress updates"""
        session_id = data.get("session_id")
        # Join room for session updates
        # In a more advanced implementation, we'd use Flask-SocketIO rooms

    def create_dashboard_html():
        """Create basic dashboard HTML template"""
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OctoBot Tentacles Agent Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .session-list {
            margin-bottom: 30px;
        }
        .session-item {
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .session-info h3 {
            margin: 0 0 5px 0;
        }
        .session-status {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-created { background: #e3f2fd; color: #1976d2; }
        .status-running { background: #fff3e0; color: #f57c00; }
        .status-completed { background: #e8f5e8; color: #388e3c; }
        .status-failed { background: #ffebee; color: #d32f2f; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            background: #4caf50;
            transition: width 0.3s ease;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-left: 5px;
        }
        .btn-primary { background: #2196f3; color: white; }
        .btn-success { background: #4caf50; color: white; }
        .btn-danger { background: #f44336; color: white; }
        .btn:hover { opacity: 0.9; }
        .create-session {
            margin-bottom: 20px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        input, select {
            padding: 8px;
            margin: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 OctoBot Tentacles Agent Dashboard</h1>
            <p>Manage and monitor tentacle testing sessions</p>
        </div>

        <div class="create-session">
            <h2>Create New Test Session</h2>
            <input type="text" id="sessionName" placeholder="Session Name" value="Test Session">
            <select id="testType">
                <option value="validation">Validation</option>
                <option value="benchmark">Benchmark</option>
                <option value="stress">Stress Test</option>
            </select>
            <button class="btn btn-primary" onclick="createSession()">Create Session</button>
        </div>

        <div class="session-list">
            <h2>Test Sessions</h2>
            <div id="sessionsList">Loading sessions...</div>
        </div>
    </div>

    <script>
        const socket = io();
        let sessions = [];

        socket.on('connect', function() {
            console.log('Connected to server');
        });

        socket.on('progress_update', function(data) {
            updateSessionProgress(data.session_id, data.progress);
        });

        function createSession() {
            const name = document.getElementById('sessionName').value;
            const testType = document.getElementById('testType').value;

            fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, test_type: testType })
            })
            .then(response => response.json())
            .then(data => {
                if (data.session_id) {
                    loadSessions();
                }
            });
        }

        function startSession(sessionId) {
            fetch(`/api/sessions/${sessionId}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ execution_mode: 'async' })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started') {
                    loadSessions();
                }
            });
        }

        function stopSession(sessionId) {
            fetch(`/api/sessions/${sessionId}/stop`, { method: 'POST' })
            .then(() => loadSessions());
        }

        function deleteSession(sessionId) {
            if (confirm('Are you sure you want to delete this session?')) {
                fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' })
                .then(() => loadSessions());
            }
        }

        function loadSessions() {
            fetch('/api/sessions')
            .then(response => response.json())
            .then(data => {
                sessions = data.sessions;
                renderSessions();
            });
        }

        function renderSessions() {
            const container = document.getElementById('sessionsList');
            container.innerHTML = '';

            if (sessions.length === 0) {
                container.innerHTML = '<p>No test sessions found.</p>';
                return;
            }

            sessions.forEach(session => {
                const div = document.createElement('div');
                div.className = 'session-item';

                const progressPercent = (session.progress * 100).toFixed(1);

                div.innerHTML = `
                    <div class="session-info">
                        <h3>${session.name}</h3>
                        <p>Type: ${session.test_type} | Created: ${new Date(session.created_at).toLocaleString()}</p>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${progressPercent}%"></div>
                        </div>
                        <small>Progress: ${progressPercent}% | Status:
                            <span class="session-status status-${session.status}">${session.status}</span>
                        </small>
                    </div>
                    <div class="session-actions">
                        ${session.status === 'created' ?
                            `<button class="btn btn-success" onclick="startSession('${session.session_id}')">Start</button>` :
                            session.status === 'running' ?
                            `<button class="btn btn-danger" onclick="stopSession('${session.session_id}')">Stop</button>` :
                            ''
                        }
                        <button class="btn btn-danger" onclick="deleteSession('${session.session_id}')">Delete</button>
                    </div>
                `;

                container.appendChild(div);
            });
        }

        function updateSessionProgress(sessionId, progress) {
            // Update progress in the UI
            const progressPercent = (progress * 100).toFixed(1);
            // Find and update the progress bar for this session
            // This would need more sophisticated DOM manipulation in a real implementation
            loadSessions(); // Simple refresh for now
        }

        // Load sessions on page load
        loadSessions();

        // Refresh sessions every 5 seconds
        setInterval(loadSessions, 5000);
    </script>
</body>
</html>
        """

        # Create templates directory if it doesn't exist
        import os

        os.makedirs("templates", exist_ok=True)

        with open("templates/dashboard.html", "w") as f:
            f.write(dashboard_html)

    # Create the dashboard template
    create_dashboard_html()

else:
    # Fallback when Flask is not available
    print("Web interface requires Flask: pip install flask flask-socketio")


def run_web_interface(host: str = "localhost", port: int = 5000, debug: bool = False):
    """
    Run the web interface

    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
    """
    if not FLASK_AVAILABLE:
        print("Cannot start web interface - Flask not available")
        return

    print(f"Starting OctoBot Tentacles Agent Dashboard on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_web_interface()
