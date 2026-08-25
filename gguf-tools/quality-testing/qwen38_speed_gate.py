#!/usr/bin/env python3
"""Qwen3.8 alias for the shared 500/15 token/s production gate."""

from qwen36_speed_gate import (  # noqa: F401
    DECODE_MIN_TOKENS_PER_SECOND,
    PREFILL_MIN_TOKENS_PER_SECOND,
    validate_speed_report,
)
