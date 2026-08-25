#!/usr/bin/env python3
"""Validate the blocking speed gate before long-context Qwen runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

from qwen36_fixtures import FixtureError, load_json


PREFILL_MIN_TOKENS_PER_SECOND = 500.0
DECODE_MIN_TOKENS_PER_SECOND = 15.0


def validate_speed_report(path: Path, *, model_sha256: str, backend: str,
                          hardware: str) -> dict:
    report = load_json(path)
    report_format = report.get("format")
    if report_format not in {
        "ds4-qwen36-speed-v1", "ds4-qwen36-speed-v2",
        "ds4-qwen38-speed-v1", "ds4-qwen38-speed-v2",
    }:
        raise FixtureError("performance report has an unsupported format")
    common_fields = ("model_sha256", "engine_commit", "build_flags", "backend",
                  "hardware", "prefill_tokens",
                  "decode_tokens", "prefill_tokens_per_second",
                  "decode_tokens_per_second", "peak_memory_mib")
    report_fields = common_fields + (("command", "context") if report_format.endswith("v1")
                                      else ("prefill_command", "decode_command",
                                            "prefill_context", "decode_context"))
    for field in report_fields:
        if field not in report:
            raise FixtureError(f"performance report is missing {field}")
    if report["model_sha256"] != model_sha256:
        raise FixtureError("performance report model SHA-256 does not match the manifest")
    if report["backend"] != backend or report["hardware"] != hardware:
        raise FixtureError("performance report backend/hardware does not match this matrix run")
    command_fields = ("command",) if report_format.endswith("v1") else (
        "prefill_command", "decode_command")
    for field in command_fields:
        if not isinstance(report[field], list) or not report[field]:
            raise FixtureError(f"performance report {field} must be a non-empty argv array")
    context_fields = ("context",) if report_format.endswith("v1") else (
        "prefill_context", "decode_context")
    for field in context_fields + ("prefill_tokens", "decode_tokens"):
        if not isinstance(report[field], int) or report[field] <= 0:
            raise FixtureError(f"performance report {field} must be a positive integer")
    for field in ("prefill_tokens_per_second", "decode_tokens_per_second", "peak_memory_mib"):
        value = report[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise FixtureError(f"performance report {field} must be a finite non-negative number")
    failures = []
    if report["prefill_tokens_per_second"] < PREFILL_MIN_TOKENS_PER_SECOND:
        failures.append(
            f"prefill {report['prefill_tokens_per_second']:.3f} tok/s < "
            f"{PREFILL_MIN_TOKENS_PER_SECOND:.0f} tok/s"
        )
    if report["decode_tokens_per_second"] < DECODE_MIN_TOKENS_PER_SECOND:
        failures.append(
            f"decode {report['decode_tokens_per_second']:.3f} tok/s < "
            f"{DECODE_MIN_TOKENS_PER_SECOND:.0f} tok/s"
        )
    if failures:
        raise FixtureError("long-context speed gate failed: " + "; ".join(failures))
    return report
