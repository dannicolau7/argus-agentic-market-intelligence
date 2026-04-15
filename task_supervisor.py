"""
task_supervisor.py — Supervised async task runner with per-task backoff and health tracking.

Replaces the flat create_task() calls in main.py lifespan.
Each background agent is registered with a name, coroutine factory, and restart policy.
On unexpected crash, the supervisor restarts the task after an exponential backoff
(30s → 60s → 120s → 300s, capped at 300s).

Health state for every task is exposed via get_health() for the /api/health endpoint.
CancelledError (clean shutdown) is NOT restarted.
"""

import asyncio
import time
from datetime import datetime
from typing import Callable, Awaitable

_tasks: dict[str, asyncio.Task]  = {}
_health: dict[str, dict]         = {}


def _now() -> str:
    return datetime.now().isoformat()


async def _supervised(name: str, factory: Callable[[], Awaitable], max_backoff: int = 300):
    """Run factory() and restart on crash with exponential backoff."""
    backoff = 30
    _health[name] = {"status": "starting", "restarts": 0, "last_error": None, "started_at": _now()}

    while True:
        try:
            _health[name]["status"]     = "running"
            _health[name]["started_at"] = _now()
            await factory()
            # Clean return (should not happen for infinite loops — treat as unexpected)
            _health[name]["status"] = "stopped"
            return

        except asyncio.CancelledError:
            _health[name]["status"] = "cancelled"
            raise   # propagate — supervisor itself was cancelled, do not restart

        except Exception as e:
            _health[name]["status"]      = "error"
            _health[name]["last_error"]  = str(e)
            _health[name]["restarts"]   += 1
            print(f"💥 [Supervisor] {name} crashed: {e}  — restarting in {backoff}s "
                  f"(restart #{_health[name]['restarts']})")
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                _health[name]["status"] = "cancelled"
                raise
            backoff = min(backoff * 2, max_backoff)


def start(name: str, factory: Callable[[], Awaitable], max_backoff: int = 300) -> asyncio.Task:
    """Register and start a supervised background task."""
    task = asyncio.create_task(_supervised(name, factory, max_backoff), name=name)
    _tasks[name] = task
    return task


async def cancel_all():
    """Cancel all supervised tasks and wait for them to finish (drain)."""
    for task in _tasks.values():
        task.cancel()
    if _tasks:
        await asyncio.gather(*_tasks.values(), return_exceptions=True)
    _tasks.clear()


def get_health() -> dict:
    """Return health snapshot for every registered task."""
    return {
        name: {
            **info,
            "alive": not (_tasks[name].done() if name in _tasks else True),
        }
        for name, info in _health.items()
    }
