"""Live Qwen server matrix for sampling, thinking, constraints, and MTP.

The live class is opt-in. ``tests/run_server_sampling_matrix.ps1`` starts the
target-only and MTP servers, supplies the environment below, and compares the
normalized outputs from both variants. The fixture class is model-free and is
safe to include in the ordinary unit suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unittest
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("DS4_SAMPLING_MATRIX_BASE_URL", "").rstrip("/")
MODEL = os.environ.get(
    "DS4_SAMPLING_MATRIX_MODEL", "Qwen3.8-27B-UD-Q4_K_S"
)
VARIANT = os.environ.get("DS4_SAMPLING_MATRIX_VARIANT", "target")
SERVER_LOG_TEXT = os.environ.get("DS4_SAMPLING_MATRIX_SERVER_LOG", "")
OUTPUT_TEXT = os.environ.get("DS4_SAMPLING_MATRIX_OUTPUT", "")
SERVER_LOG = Path(SERVER_LOG_TEXT)
OUTPUT = Path(OUTPUT_TEXT)
HTTP_TIMEOUT_SECONDS = 900
SEED = 424242
PHASE_PREFIX = "ds4-server: phase profile "


SAMPLING_CASES = (
    {
        "id": "greedy_nothink",
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 20,
        "thinking": False,
    },
    {
        "id": "sampled_nothink",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "thinking": False,
    },
    {
        "id": "greedy_think",
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 20,
        "thinking": True,
    },
    {
        "id": "sampled_think",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "thinking": True,
    },
)

SCENARIOS = ("plain", "optional_text", "required_tool")
OPTIONAL_TEXT = "MATRIX_OK_PLAIN_RESPONSE_WITHOUT_TOOL_CALLS_1234567890"

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "matrix_probe",
        "description": "Record the fixed sampling-matrix probe.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "string", "const": "matrix-ok"},
                "count": {"type": "integer", "const": 2},
            },
            "required": ["value", "count"],
            "additionalProperties": False,
        },
    },
}

PROBE_RESPONSES_TOOL = {
    "type": "function",
    "name": PROBE_TOOL["function"]["name"],
    "description": PROBE_TOOL["function"]["description"],
    "parameters": PROBE_TOOL["function"]["parameters"],
}


def request_payload(scenario: str, sampling: dict[str, Any]) -> dict[str, Any]:
    """Build a Chat or Responses request without hiding a matrix dimension."""
    thinking = bool(sampling["thinking"])
    if scenario == "plain":
        prompt = "Reply with exactly MATRIX_OK and nothing else."
    elif scenario == "optional_text":
        prompt = (
            f"Do not call any tool. Reply with exactly {OPTIONAL_TEXT} and "
            "nothing else."
        )
    elif scenario == "required_tool":
        prompt = (
            "Call matrix_probe exactly once with value matrix-ok and count 2. "
            "Do not emit any other call or public prose."
        )
    else:
        raise ValueError(f"unknown sampling-matrix scenario: {scenario}")

    if scenario == "plain":
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "top_k": sampling["top_k"],
            "min_p": 0.0,
            "seed": SEED,
            "max_tokens": 768 if thinking else 384,
            "stream": False,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }

    # DS4 intentionally accepts its agentic capability extension only through
    # /v1/responses.  Unlike the Chat endpoint, Responses currently exposes
    # temperature and top_p but not the non-standard top_k/min_p/seed knobs.
    payload = {
        "model": MODEL,
        "input": [{"role": "user", "content": prompt}],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "max_output_tokens": 768 if thinking else 384,
        "stream": False,
        "reasoning": {
            "effort": "high" if thinking else "none",
            "summary": "auto" if thinking else "none",
        },
        "tools": [PROBE_RESPONSES_TOOL],
        "agentic": {
            "allowed_tools": ["matrix_probe"],
            "allowed_skills": [],
        },
    }
    if scenario == "required_tool":
        payload["tool_choice"] = "required"
    return payload


def post_inference(scenario: str, payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = "chat/completions" if scenario == "plain" else "responses"
    request = Request(
        f"{BASE_URL}/v1/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"server returned HTTP {exc.code}: {detail}") from exc


def parse_inference(response: dict[str, Any], scenario: str,
                    sampling: dict[str, Any]) -> dict[str, Any]:
    thinking = bool(sampling["thinking"])
    if scenario != "plain":
        return parse_responses(response, scenario, sampling)

    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"invalid chat choices: {response!r}")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise AssertionError(f"missing assistant message: {response!r}")
    finish = choice.get("finish_reason")
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = ""
    if not isinstance(reasoning, str):
        raise AssertionError(f"invalid reasoning content: {response!r}")
    if thinking and not reasoning.strip():
        raise AssertionError(f"thinking enabled but reasoning is empty: {response!r}")
    if not thinking and reasoning.strip():
        raise AssertionError(f"thinking disabled but reasoning was returned: {response!r}")

    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise AssertionError(f"invalid assistant content: {response!r}")
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise AssertionError(f"invalid tool_calls: {response!r}")
    calls: list[dict[str, Any]] = []
    for call in raw_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            raise AssertionError(f"invalid tool call: {call!r}")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise AssertionError(f"invalid tool arguments: {call!r}")
        calls.append({
            "name": function.get("name"),
            "arguments": json.loads(arguments),
        })

    if scenario in {"plain", "optional_text"}:
        if finish != "stop" or calls:
            raise AssertionError(
                f"{scenario} expected text/stop without calls: {response!r}"
            )
        expected_content = "MATRIX_OK" if scenario == "plain" else OPTIONAL_TEXT
        if content.strip() != expected_content:
            raise AssertionError(
                f"{scenario} returned unexpected public text: {content!r}"
            )
    else:
        expected = [{
            "name": "matrix_probe",
            "arguments": {"value": "matrix-ok", "count": 2},
        }]
        if finish != "tool_calls" or calls != expected or content.strip():
            raise AssertionError(
                f"required tool result differs from contract: {response!r}"
            )

    return normalized_result(
        finish=finish,
        reasoning=reasoning,
        content=content,
        calls=calls,
        usage=response.get("usage"),
    )


def parse_responses(response: dict[str, Any], scenario: str,
                    sampling: dict[str, Any]) -> dict[str, Any]:
    thinking = bool(sampling["thinking"])
    sampled = float(sampling["temperature"]) > 0.0
    output = response.get("output")
    if not isinstance(output, list):
        raise AssertionError(f"missing Responses output: {response!r}")
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    calls: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            raise AssertionError(f"invalid Responses output item: {item!r}")
        item_type = item.get("type")
        if item_type == "reasoning":
            summaries = item.get("summary") or []
            if not isinstance(summaries, list):
                raise AssertionError(f"invalid reasoning summary: {item!r}")
            reasoning_parts.extend(
                part["text"] for part in summaries
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        elif item_type == "message":
            parts = item.get("content") or []
            if not isinstance(parts, list):
                raise AssertionError(f"invalid message content: {item!r}")
            content_parts.extend(
                part["text"] for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        elif item_type in {"function_call", "custom_tool_call"}:
            arguments = item.get("arguments")
            if not isinstance(item.get("name"), str) or not isinstance(arguments, str):
                raise AssertionError(f"invalid function call: {item!r}")
            calls.append({
                "name": item["name"],
                "arguments": json.loads(arguments),
            })

    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    if thinking and not reasoning.strip():
        raise AssertionError(f"thinking enabled but reasoning is empty: {response!r}")
    if not thinking and reasoning.strip():
        raise AssertionError(f"thinking disabled but reasoning was returned: {response!r}")
    finish = "tool_calls" if calls else "stop"
    if scenario == "optional_text":
        expected_text = content.strip() == OPTIONAL_TEXT
        # At non-zero temperature this is intentionally an unconstrained
        # ordinary-text branch: requiring verbatim text would test model luck,
        # not the SEARCH/tool-call safety boundary.  Greedy remains exact.
        sampled_text = sampled and bool(content.strip())
        if calls or not (expected_text or sampled_text):
            raise AssertionError(
                f"optional agentic response differs from contract: {response!r}"
            )
    else:
        expected = [{
            "name": "matrix_probe",
            "arguments": {"value": "matrix-ok", "count": 2},
        }]
        if calls != expected or content.strip():
            raise AssertionError(
                f"required agentic tool result differs from contract: {response!r}"
            )
    if response.get("status") not in {None, "completed"}:
        raise AssertionError(f"Responses request did not complete: {response!r}")
    return normalized_result(
        finish=finish,
        reasoning=reasoning,
        content=content,
        calls=calls,
        usage=response.get("usage"),
        contract_content="PUBLIC_TEXT" if scenario == "optional_text" and sampled else None,
    )


def normalized_result(*, finish: Any, reasoning: str, content: str,
                      calls: list[dict[str, Any]], usage: Any,
                      contract_content: str | None = None) -> dict[str, Any]:
    semantic = {
        "finish_reason": finish,
        "reasoning_content": reasoning,
        "content": content,
        "tool_calls": calls,
    }
    contract = {
        "finish_reason": finish,
        "thinking_present": bool(reasoning.strip()),
        "content": content.strip() if contract_content is None else contract_content,
        "tool_calls": calls,
    }
    encoded = json.dumps(
        semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    contract_encoded = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "semantic": semantic,
        "semantic_sha256": hashlib.sha256(encoded).hexdigest(),
        "contract_sha256": hashlib.sha256(contract_encoded).hexdigest(),
        "usage": usage,
    }


def read_phase_segment(offset: int) -> str:
    deadline = time.monotonic() + 30.0
    segment = ""
    while time.monotonic() < deadline:
        segment = SERVER_LOG.read_bytes()[offset:].decode(
            "utf-8", errors="replace"
        )
        if PHASE_PREFIX in segment:
            return segment
        time.sleep(0.1)
    raise AssertionError("server did not publish a phase profile for the request")


def phase_counters(segment: str) -> dict[str, int]:
    lines = [line for line in segment.splitlines() if PHASE_PREFIX in line]
    if not lines:
        raise AssertionError("missing phase profile line")
    fields = (
        "filter_calls",
        "search_static_steps",
        "search_static_fallbacks",
        "mtp_constrained_target_only_steps",
        "mtp_constrained_speculative_cycles",
        "mtp_constrained_sampled_rows",
        "mtp_constrained_accepted_drafts",
        "mtp_constrained_fallbacks",
    )
    counters: dict[str, int] = {}
    for field in fields:
        match = re.search(rf"(?:^| ){field}=(\d+)(?: |$)", lines[-1])
        if match is None:
            raise AssertionError(f"phase profile omitted {field}: {lines[-1]}")
        counters[field] = int(match.group(1))
    counters["mtp_cycles"] = segment.count("QWEN_MTP_CYCLE")
    return counters


class SamplingMatrixFixtureTests(unittest.TestCase):
    """Model-free checks that keep every requested matrix axis explicit."""

    def test_matrix_covers_temperature_and_thinking_cross_product(self) -> None:
        dimensions = {
            (float(case["temperature"]) == 0.0, bool(case["thinking"]))
            for case in SAMPLING_CASES
        }
        self.assertEqual(
            dimensions,
            {(True, False), (False, False), (True, True), (False, True)},
        )

    def test_payloads_cover_plain_optional_and_required_surfaces(self) -> None:
        sampling = SAMPLING_CASES[1]
        plain = request_payload("plain", sampling)
        optional = request_payload("optional_text", sampling)
        required = request_payload("required_tool", sampling)
        self.assertNotIn("tools", plain)
        self.assertIn("messages", plain)
        self.assertIn("seed", plain)
        self.assertIn("tools", optional)
        self.assertIn("input", optional)
        self.assertNotIn("seed", optional)
        self.assertEqual(
            optional["agentic"],
            {"allowed_tools": ["matrix_probe"], "allowed_skills": []},
        )
        self.assertNotIn("tool_choice", optional)
        self.assertEqual(required["tool_choice"], "required")
        self.assertEqual(plain["seed"], SEED)

    def test_phase_counters_require_constrained_mtp_telemetry(self) -> None:
        counters = phase_counters(
            "ds4-server: phase profile ctx=1..5 gen=4 mode=optimized "
            "filter_calls=12 search_static_steps=3 search_static_fallbacks=0 "
            "mtp_constrained_target_only_steps=2 "
            "mtp_constrained_speculative_cycles=4 "
            "mtp_constrained_sampled_rows=7 "
            "mtp_constrained_accepted_drafts=2 "
            "mtp_constrained_fallbacks=0\n"
            "QWEN_MTP_CYCLE accepted=2\n"
        )
        self.assertEqual(counters["mtp_constrained_speculative_cycles"], 4)
        self.assertEqual(counters["mtp_constrained_sampled_rows"], 7)
        self.assertEqual(counters["mtp_constrained_accepted_drafts"], 2)
        self.assertEqual(counters["mtp_constrained_fallbacks"], 0)
        self.assertEqual(counters["mtp_cycles"], 1)

    def test_phase_counters_fail_closed_when_new_telemetry_is_missing(self) -> None:
        with self.assertRaisesRegex(
            AssertionError, "mtp_constrained_speculative_cycles"
        ):
            phase_counters(
                "ds4-server: phase profile filter_calls=1 "
                "search_static_steps=0 search_static_fallbacks=0 "
                "mtp_constrained_target_only_steps=1\n"
            )

    def test_sampled_optional_text_compares_the_public_contract(self) -> None:
        def response(text: str) -> dict[str, Any]:
            return {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }],
                "usage": {},
            }

        sampled = SAMPLING_CASES[1]
        first = parse_responses(response("MATRIX_OK"), "optional_text", sampled)
        second = parse_responses(
            response("A different but valid public answer."),
            "optional_text",
            sampled,
        )
        self.assertNotEqual(first["semantic_sha256"], second["semantic_sha256"])
        self.assertEqual(first["contract_sha256"], second["contract_sha256"])

        with self.assertRaises(AssertionError):
            parse_responses(
                response("A non-verbatim greedy answer."),
                "optional_text",
                SAMPLING_CASES[0],
            )


@unittest.skipUnless(BASE_URL, "run through tests/run_server_sampling_matrix.ps1")
class SamplingMatrixLiveTests(unittest.TestCase):
    """Exercise the complete server matrix against one loaded variant."""

    def test_full_matrix(self) -> None:
        if VARIANT not in {"target", "mtp"}:
            self.fail(f"invalid DS4_SAMPLING_MATRIX_VARIANT: {VARIANT}")
        if not SERVER_LOG_TEXT or not SERVER_LOG.is_file():
            self.fail(f"server log does not exist: {SERVER_LOG}")
        if not OUTPUT_TEXT:
            self.fail("DS4_SAMPLING_MATRIX_OUTPUT is required")

        results: list[dict[str, Any]] = []
        for sampling in SAMPLING_CASES:
            for scenario in SCENARIOS:
                with self.subTest(sampling=sampling["id"], scenario=scenario):
                    offset = SERVER_LOG.stat().st_size
                    response = post_inference(
                        scenario, request_payload(scenario, sampling)
                    )
                    parsed = parse_inference(response, scenario, sampling)
                    segment = read_phase_segment(offset)
                    counters = phase_counters(segment)

                    if scenario == "plain":
                        self.assertEqual(counters["filter_calls"], 0)
                        if VARIANT == "mtp":
                            self.assertGreater(counters["mtp_cycles"], 0)
                        else:
                            self.assertEqual(counters["mtp_cycles"], 0)
                    else:
                        self.assertGreater(counters["filter_calls"], 0)
                        if VARIANT == "mtp":
                            if scenario == "required_tool":
                                if sampling["thinking"]:
                                    # MTP accelerates private reasoning, then
                                    # stops at the required DSML boundary where
                                    # live sweeps show target-only is faster.
                                    self.assertGreater(
                                        counters[
                                            "mtp_constrained_speculative_cycles"
                                        ],
                                        0,
                                    )
                                else:
                                    self.assertEqual(
                                        counters[
                                            "mtp_constrained_speculative_cycles"
                                        ],
                                        0,
                                    )
                                self.assertEqual(
                                    counters["mtp_constrained_sampled_rows"], 0
                                )
                            else:
                                self.assertGreater(
                                    counters[
                                        "mtp_constrained_speculative_cycles"
                                    ],
                                    0,
                                )
                                self.assertGreater(
                                    counters["mtp_constrained_sampled_rows"], 0
                                )
                        else:
                            self.assertEqual(
                                counters["mtp_constrained_speculative_cycles"], 0
                            )
                            self.assertEqual(
                                counters["mtp_constrained_sampled_rows"], 0
                            )
                    if scenario == "optional_text":
                        if sampling["thinking"]:
                            # The retained SEARCH suffix still contains the
                            # closing </think> marker. Target-only decoding
                            # sees a dynamic frontier before SEARCH becomes
                            # static; one MTP verifier window may cross that
                            # boundary entirely in its temporary parser state.
                            if VARIANT == "target":
                                self.assertGreater(
                                    counters["search_static_fallbacks"], 0
                                )
                            if float(sampling["temperature"]) == 0.0:
                                self.assertGreater(
                                    counters["search_static_steps"], 0
                                )
                        else:
                            self.assertGreater(
                                counters["search_static_steps"], 0
                            )
                            self.assertEqual(
                                counters["search_static_fallbacks"], 0
                            )
                    if scenario == "required_tool" and VARIANT == "mtp":
                        # The adaptive gate stops before required DSML; it must
                        # not enter the exhaustive verifier-mask fallback.
                        self.assertEqual(
                            counters["mtp_constrained_fallbacks"], 0
                        )

                    results.append({
                        "id": f"{sampling['id']}::{scenario}",
                        "sampling": sampling,
                        "scenario": scenario,
                        "api": "chat_completions" if scenario == "plain" else "responses",
                        **parsed,
                        "phase_counters": counters,
                    })

        record = {
            "schema_version": 1,
            "variant": VARIANT,
            "model": MODEL,
            "seed": SEED,
            "status": "PASS",
            "cases": results,
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        if OUTPUT.exists():
            self.fail(f"refusing to overwrite sampling matrix: {OUTPUT}")
        OUTPUT.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
