"""In-band restart scheduling, kept separate from shutdown teardown phases."""

import asyncio
import logging
from contextvars import Context

logger = logging.getLogger("gateway.run")


class GatewayRestartControlMixin:
    """Schedule one host restart and let explicit force escalate its existing waits."""

    def request_restart(self, *, detached: bool = False, via_service: bool = False, force: bool = False) -> bool:
        if force and not getattr(self, "_restart_force", False):
            self._restart_force = True
            logger.warning("Forced gateway restart requested; skipping active-work waits")
            if self._restart_task_started:
                return True  # The existing wait observes the upgrade; never start a second teardown.
        if self._restart_task_started:
            return False
        self._restart_requested = True
        self._restart_detached = detached
        self._restart_via_service = via_service
        self._restart_task_started = True
        # Refuse new turns; keep ``_running`` True so the active turn can still deliver its final response.
        self._draining = True
        from gateway.drain_control import open_restart_wait
        open_restart_wait(self)  # clients may still answer a turn parked on approval/clarify (#139041)
        # The restart's after-turn wait is a drain window too: pollers of GET /v1/runs/{id} must see
        # the boundary from the moment new turns are refused, not only once stop() begins (#115133).
        self._mark_api_runs_shutdown_requested()

        async def _run_restart() -> None:
            if not getattr(self, "_restart_force", False):
                await self._await_active_work_before_restart()
            # Detached helper only AFTER the after-turn wait, or its drain_timeout+5 deadline fires mid-turn.
            if detached:
                from gateway.run_shutdown import _log_suppressed
                with _log_suppressed(logging.ERROR, "Failed to launch detached gateway restart helper: %s"):
                    await self._launch_detached_restart_command()
            await asyncio.sleep(0.05)
            await self.stop(restart=True, detached_restart=detached, service_restart=via_service)

        # NOT in _background_tasks: _stop_impl cancels those, which would cancel _run_restart while it
        # awaits _stop_task and skip _shutdown_event.set() / _exit_code = 75 (#12875). We still hold a strong reference in
        # self._restart_task: a bare asyncio.create_task() keeps only a weak reference, so the event loop
        # may garbage-collect a still-pending task mid-flight. The cancel loop in _stop_impl explicitly
        # skips _restart_task for the same reason it skips _stop_task.
        # Empty Context: /restart is handled inside the requester's profile scope, and a copied context
        # would run the HOST restart as that profile (watcher HERMES_HOME, stop()'s flushes).
        self._restart_task = Context().run(lambda: asyncio.create_task(_run_restart()))
        return True
