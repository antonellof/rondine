"""Managed server process lifecycle."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rondine.engines.base import LaunchSpec
from rondine.paths import run_dir


@dataclass
class RunRecord:
    pid: int
    engine: str
    model_id: str
    alias: str
    host: str
    port: int
    argv: list[str]
    log_path: str
    started_at: float


def _record_path(name: str = "default") -> Path:
    return run_dir() / f"{name}.json"


def _log_path(name: str = "default") -> Path:
    return run_dir() / f"{name}.log"


def start_server(spec: LaunchSpec, *, name: str = "default", foreground: bool = False) -> RunRecord:
    log = _log_path(name)
    if foreground:
        # Caller owns the process; still write a marker
        proc = subprocess.Popen(spec.argv, env=spec.env, cwd=spec.cwd)
        record = RunRecord(
            pid=proc.pid,
            engine=spec.engine,
            model_id=spec.model_id,
            alias=spec.alias,
            host=spec.host,
            port=spec.port,
            argv=list(spec.argv),
            log_path=str(log),
            started_at=time.time(),
        )
        _record_path(name).write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        proc.wait()
        return record

    log_fh = log.open("ab")
    proc = subprocess.Popen(
        spec.argv,
        env=spec.env,
        cwd=spec.cwd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    record = RunRecord(
        pid=proc.pid,
        engine=spec.engine,
        model_id=spec.model_id,
        alias=spec.alias,
        host=spec.host,
        port=spec.port,
        argv=list(spec.argv),
        log_path=str(log),
        started_at=time.time(),
    )
    _record_path(name).write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return record


def load_record(name: str = "default") -> RunRecord | None:
    path = _record_path(name)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord(**raw)


def is_running(record: RunRecord) -> bool:
    try:
        os.kill(record.pid, 0)
    except OSError:
        return False
    return True


def stop_server(name: str = "default", timeout: float = 15.0) -> dict[str, Any]:
    record = load_record(name)
    if record is None:
        return {"ok": False, "error": f"no run record named '{name}'"}
    if not is_running(record):
        _record_path(name).unlink(missing_ok=True)
        return {"ok": True, "message": "process already stopped", "pid": record.pid}
    try:
        os.kill(record.pid, signal.SIGTERM)
    except OSError as exc:
        return {"ok": False, "error": str(exc), "pid": record.pid}
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_running(record):
            _record_path(name).unlink(missing_ok=True)
            return {"ok": True, "message": "stopped", "pid": record.pid}
        time.sleep(0.2)
    try:
        os.kill(record.pid, signal.SIGKILL)
    except OSError:
        pass
    _record_path(name).unlink(missing_ok=True)
    return {"ok": True, "message": "killed", "pid": record.pid}
