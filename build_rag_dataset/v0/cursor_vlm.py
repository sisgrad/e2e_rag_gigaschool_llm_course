"""Shared Cursor SDK invocation helpers for the v0 VLM stages."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import mimetypes
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["producer"]["model"] == config["verifier"]["model"]:
        raise ValueError("Producer and verifier models must differ")
    return config


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _model_selection(role_config: dict[str, Any]):
    from cursor_sdk import ModelParameterValue, ModelSelection

    parameters = [
        ModelParameterValue(id=str(key), value=str(value))
        for key, value in (role_config.get("parameters") or {}).items()
    ]
    return ModelSelection(id=role_config["model"], params=parameters)


def _user_message(image_path: Path, prompt: str):
    from cursor_sdk import SDKImage, UserMessage

    mime_type = (
        mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    )
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return UserMessage(
        text=prompt,
        images=[SDKImage(data=encoded, mime_type=mime_type)],
    )


def _usage_dict(usage: object | None) -> Any:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if hasattr(usage, "__dict__"):
        return json.loads(json.dumps(vars(usage), default=str))
    return str(usage)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_cursor_json(
    *,
    role: str,
    image_path: Path,
    prompt: str,
    config: dict[str, Any],
    cache_root: Path,
    cache_namespace: str,
    workspace: Path,
) -> dict[str, Any]:
    """Run one local Cursor agent with an image and parse its JSON result."""
    from cursor_sdk import (
        Agent,
        AgentOptions,
        CursorAgentError,
        LocalAgentOptions,
    )

    if role not in {"producer", "verifier"}:
        raise ValueError(f"Unknown VLM role: {role}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    api_key = os.getenv("CURSOR_API_KEY")
    if not api_key:
        raise RuntimeError("CURSOR_API_KEY is not set")

    role_config = config[role]
    execution = config.get("execution", {})
    cache_key = stable_hash(
        {
            "namespace": cache_namespace,
            "role": role,
            "model": role_config,
            "image_size": image_path.stat().st_size,
            "image_mtime_ns": image_path.stat().st_mtime_ns,
            "prompt": prompt,
            "sdk_version": importlib.metadata.version("cursor-sdk"),
        }
    )
    cache_path = cache_root / role / f"{cache_key}.json"
    if execution.get("cache", True) and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("data") is not None and not cached.get("error"):
            cached["cache_hit"] = True
            return cached

    attempts: list[dict[str, Any]] = []
    final_error: Exception | None = None
    max_retries = int(execution.get("max_retries", 3))
    base_sleep = float(execution.get("retry_base_seconds", 5))
    for attempt in range(1, max_retries + 1):
        started = time.monotonic()
        try:
            result = Agent.prompt(
                _user_message(image_path, prompt),
                AgentOptions(
                    api_key=api_key,
                    model=_model_selection(role_config),
                    local=LocalAgentOptions(
                        cwd=workspace,
                        setting_sources=execution.get(
                            "setting_sources", []
                        ),
                    ),
                    name=f"video-retrieval-{role}",
                ),
            )
            elapsed_ms = int((time.monotonic() - started) * 1000)
            attempt_row = {
                "attempt": attempt,
                "agent_id": result.agent_id,
                "run_id": result.id,
                "status": str(result.status),
                "duration_ms": result.duration_ms or elapsed_ms,
                "usage": _usage_dict(result.usage),
            }
            attempts.append(attempt_row)
            if str(result.status) != "finished":
                final_error = RuntimeError(
                    f"Cursor run status={result.status}"
                )
            else:
                try:
                    data = extract_json_object(result.result)
                except Exception as error:
                    final_error = error
                    attempt_row["parse_error"] = (
                        f"{type(error).__name__}:{error}"
                    )
                else:
                    payload = {
                        "schema_version": "cursor-vlm-result-1",
                        "role": role,
                        "model": role_config["model"],
                        "sdk_version": importlib.metadata.version(
                            "cursor-sdk"
                        ),
                        "created_at": utc_now(),
                        "cache_key": cache_key,
                        "cache_hit": False,
                        "attempts": attempts,
                        "raw_result": result.result,
                        "data": data,
                    }
                    dump_json(cache_path, payload)
                    return payload
        except CursorAgentError as error:
            final_error = error
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "startup_error",
                    "error": str(error),
                    "retryable": getattr(error, "is_retryable", False),
                    "retry_after": str(
                        getattr(error, "retry_after", "") or ""
                    ),
                }
            )
            if not getattr(error, "is_retryable", False):
                break
        except Exception as error:
            final_error = error
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "exception",
                    "error": f"{type(error).__name__}:{error}",
                }
            )
        if attempt < max_retries:
            time.sleep(
                base_sleep * (2 ** (attempt - 1)) + random.random()
            )

    failure = {
        "schema_version": "cursor-vlm-result-1",
        "role": role,
        "model": role_config["model"],
        "sdk_version": importlib.metadata.version("cursor-sdk"),
        "created_at": utc_now(),
        "cache_key": cache_key,
        "cache_hit": False,
        "attempts": attempts,
        "error": (
            f"{type(final_error).__name__}:{final_error}"
            if final_error
            else "unknown"
        ),
        "data": None,
    }
    dump_json(cache_path, failure)
    return failure
