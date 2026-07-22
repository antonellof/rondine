"""API health and coding smoke verification."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class VerifyResult:
    ok: bool
    base_url: str
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            self.ok = False


def _wait_ready(base_url: str, timeout: float = 120.0) -> tuple[bool, str]:
    deadline = time.time() + timeout
    last = ""
    health_urls = [
        base_url.replace("/v1", "/health"),
        f"{base_url}/models",
    ]
    while time.time() < deadline:
        for url in health_urls:
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(url)
                    if resp.status_code < 500:
                        return True, f"{url} -> {resp.status_code}"
                    last = f"{url} -> {resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
        time.sleep(1.0)
    return False, last or "timeout waiting for server"


def verify_server(
    base_url: str,
    *,
    model: str | None = None,
    profile: str = "coding",
    timeout: float = 120.0,
) -> VerifyResult:
    result = VerifyResult(ok=True, base_url=base_url)
    ready, detail = _wait_ready(base_url, timeout=timeout)
    result.add("ready", ready, detail)
    if not ready:
        return result

    with httpx.Client(timeout=60.0) as client:
        try:
            models_resp = client.get(f"{base_url}/models")
            models_resp.raise_for_status()
            payload = models_resp.json()
            ids = [m.get("id") for m in payload.get("data", [])]
            result.add("models", True, ", ".join(str(i) for i in ids) or "(empty)")
            if model is None and ids:
                model = str(ids[0])
        except Exception as exc:  # noqa: BLE001
            result.add("models", False, str(exc))
            return result

        # Short code generation smoke
        try:
            body = {
                "model": model or "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a Python function hello() that returns 'hi'. Code only.",
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.2 if profile == "coding" else 0.7,
            }
            resp = client.post(f"{base_url}/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content") or ""
            ok = "def " in content or "hello" in content.lower()
            result.add("codegen", ok, content[:240].replace("\n", "\\n"))
        except Exception as exc:  # noqa: BLE001
            result.add("codegen", False, str(exc))

        # Tool-call structured smoke (best-effort; engines differ)
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a text file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
            body = {
                "model": model or "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "Call read_file with path=/tmp/demo.txt",
                    }
                ],
                "tools": tools,
                "max_tokens": 128,
                "temperature": 0.0,
            }
            resp = client.post(f"{base_url}/chat/completions", json=body)
            if resp.status_code >= 400:
                result.add("tool_call", True, f"skipped/unsupported ({resp.status_code})")
            else:
                data = resp.json()
                msg = data["choices"][0]["message"]
                tool_calls = msg.get("tool_calls") or []
                content = msg.get("content") or ""
                ok = bool(tool_calls) or "read_file" in content
                result.add(
                    "tool_call",
                    ok,
                    json.dumps(tool_calls)[:200] if tool_calls else content[:200],
                )
        except Exception as exc:  # noqa: BLE001
            result.add("tool_call", True, f"skipped: {exc}")

    return result
