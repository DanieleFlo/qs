#!/usr/bin/env python3
"""Profile real /agent story turns and attribute DSML free-text cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from agent_wiki.backend.core.agent.deep_agent import DeepAgent
from agent_wiki.backend.core.client import Client
from agent_wiki.backend.core.config.hds_config import HDSGraphConfig, HDSNodeConfig
from agent_wiki.backend.providers.agentic_openai import _AgenticOpenAIProvider
from agent_wiki.configs.runtime import (
    LoopConfigOverride,
    ModelRequestConfigOverride,
    RuntimeConfigOverrides,
)
from agent_wiki.configs.settings import SystemConfig
from agent_wiki.wiki import WikiGraphNodeConfig

import perf_harness


TOOL_PAYLOAD_PROMPT = (
    "Genera una storia in italiano di esattamente 300 parole. "
    "Usa la procedura exit-with-info configurata: attiva l'exit skill e inserisci "
    "l'intera storia nel campo message di exit-with-info-tool, usando la forma "
    "{{null}} richiesta dal tool. Non usare altri skill o tool, non leggere o "
    "modificare file e non aggiungere commenti alla storia."
)

PLAIN_STORY_PROMPT = (
    "Questo è un benchmark deterministico di generazione testuale. Scrivi "
    "direttamente una semplice storia in italiano tra 250 e 300 parole. Rispondi "
    "con la sola storia come testo normale: non usare tool, skill o procedure di "
    "uscita, non leggere o modificare file e non aggiungere introduzioni, conteggi "
    "o commenti."
)

SCENARIO_PROMPTS = {
    "tool-payload": TOOL_PAYLOAD_PROMPT,
    "plain-story": PLAIN_STORY_PROMPT,
}

DSML_MARKERS = (
    "<｜DSML｜tool_calls>",
    "<｜DSML｜invoke",
    "<|DSML|tool_calls>",
    "<|DSML|invoke",
)


class ExplicitThinkingAgenticProvider(_AgenticOpenAIProvider):
    """Make the disabled-thinking request explicit on the Responses wire."""

    def build_request(self, **kwargs: Any) -> dict[str, Any]:
        request = super().build_request(**kwargs)
        if not bool(kwargs.get("thinking", False)):
            request["reasoning"] = {"effort": "none"}
        return request


def make_agent(
    *, client: Client, workroot: Path, max_output_tokens: int
) -> DeepAgent:
    """Build the packaged bootstrap-wiki graph used by the /agent frontend."""
    runtime = RuntimeConfigOverrides(
        model=ModelRequestConfigOverride(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            thinking=False,
        ),
        loop=LoopConfigOverride(max_steps=4),
    )
    graph = HDSGraphConfig(
        system_config=SystemConfig.load_default(),
        main=HDSNodeConfig(
            name="bootstrap-wiki",
            client=client,
            workroot=workroot,
            runtime=runtime,
            children=[WikiGraphNodeConfig()],
        ),
    )
    return DeepAgent(node_config=graph)


def context_tokens(span: str) -> int:
    match = re.fullmatch(r"\d+\.\.(\d+):\d+", span)
    if match is None:
        raise ValueError(f"unrecognized phase context span: {span}")
    return int(match.group(1))


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\wÀ-ÖØ-öø-ÿ’'-]+\b", text, flags=re.UNICODE))


def phase_attribution(phase: dict[str, object]) -> dict[str, object]:
    wall_ms = float(phase["wall_ms"])
    phase_names = (
        "forced_prefix_probe_ms",
        "sampling_mask_build_ms",
        "forced_sync_ms",
        "filtered_sample_ms",
        "eval_ms",
        "residual_ms",
    )
    milliseconds = {
        name: float(phase.get(name, 0.0))
        for name in phase_names
    }
    return {
        "milliseconds": milliseconds,
        "wall_percent": {
            name: value * 100.0 / wall_ms if wall_ms > 0.0 else 0.0
            for name, value in milliseconds.items()
        },
    }


def read_new_log(path: Path, offset: int, timeout: float = 60.0) -> str:
    """Wait for phase metrics and every published response checkpoint."""
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        text = path.read_bytes()[offset:].decode("utf-8", errors="replace")
        finishes = len(re.findall(
            r" finish=(?:stop|tool_calls|length|error)", text
        ))
        checkpoints = text.count(
            "post-response checkpoint rebuilt source=memory-request"
        )
        if (
            "ds4-server: phase profile " in text
            and finishes > 0
            and checkpoints >= finishes
        ):
            return text
        time.sleep(0.1)
    return text


def history_tool_calls(history: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in history:
        for call in getattr(message, "tool_calls", ()) or ():
            calls.append(dict(call))
    return calls


def post_response_status(segment: str) -> dict[str, object]:
    finishes = [
        match.start()
        for match in re.finditer(
            r" finish=(?:stop|tool_calls|length|error)", segment
        )
    ]
    rebuild_pattern = re.compile(
        r"post-response checkpoint rebuilt source=memory-request "
        r"cached=(\d+) response_prefill=(\d+) target=(\d+) location=(\w+)",
    )
    memory_matches = list(rebuild_pattern.finditer(segment))
    memory_rebuilds = [match.groups() for match in memory_matches]
    valid_rebuilds = all(
        int(cached) + int(prefill) == int(target) and int(prefill) > 0
        for cached, prefill, target, _location in memory_rebuilds
    )
    disk_reload_after_finish = False
    for finish in finishes:
        rebuild = next(
            (match for match in memory_matches if match.start() > finish), None
        )
        window = segment[finish:rebuild.end() if rebuild else len(segment)]
        if "source=disk" in window or "kv cache hit text" in window:
            disk_reload_after_finish = True
            break
    return {
        "finish_count": len(finishes),
        "memory_request_rebuilds": len(memory_rebuilds),
        "valid_memory_request_rebuilds": valid_rebuilds,
        "locations": sorted({row[3] for row in memory_rebuilds}),
        "disk_reload_after_finish": disk_reload_after_finish,
    }


def run_once(
    *,
    client: Client,
    workroot: Path,
    server_log: Path,
    scenario: str,
    prompt: str,
    max_output_tokens: int,
    minimum_context: int,
    minimum_output_tokens: int,
    minimum_words: int,
    measured_index: int,
    warmup: bool,
) -> dict[str, object]:
    agent = make_agent(
        client=client,
        workroot=workroot,
        max_output_tokens=max_output_tokens,
    )
    offset = server_log.stat().st_size
    started = time.monotonic()
    result = agent.run(user_message=prompt)
    request_wall_ms = (time.monotonic() - started) * 1000.0
    segment = read_new_log(server_log, offset)
    phases = perf_harness.parse_server_phase_profiles(segment)
    constrained_phases = [
        phase for phase in phases
        if int(phase.get("filter_calls", 0)) > 0
    ]
    if not constrained_phases:
        raise RuntimeError("agent turn emitted no constrained phase profile")
    story_phase = max(
        constrained_phases,
        key=lambda phase: int(phase["generation_tokens"]),
    )
    occupied = context_tokens(str(story_phase["context_span"]))
    generation = int(story_phase["generation_tokens"])
    wall_ms = float(story_phase["wall_ms"])
    response = result.response
    words = word_count(response)
    tool_calls = history_tool_calls(result.chat_history)
    generation_error = re.search(
        r"finish=error[^\r\n]*error=\"([^\"]+)\"", segment
    )
    finish_stop = bool(re.search(r" finish=stop", segment))
    protocol_marker_present = any(marker in response for marker in DSML_MARKERS)
    checkpoint = post_response_status(segment)
    failures: list[str] = []
    if generation_error:
        failures.append(generation_error.group(1))
    if occupied < minimum_context:
        failures.append(
            f"context {occupied} is below minimum {minimum_context}"
        )
    if not checkpoint["memory_request_rebuilds"]:
        failures.append("post-response memory-request checkpoint is missing")
    if not checkpoint["valid_memory_request_rebuilds"]:
        failures.append("post-response checkpoint token accounting is invalid")
    if checkpoint["disk_reload_after_finish"]:
        failures.append("post-response checkpoint reloaded state from SSD")
    if scenario == "plain-story":
        if not finish_stop:
            failures.append("plain story did not finish with stop")
        if result.token_usage.request_count != 1:
            failures.append(
                "plain story used "
                f"{result.token_usage.request_count} model requests instead of one"
            )
        if tool_calls:
            failures.append(f"plain story emitted {len(tool_calls)} tool calls")
        if protocol_marker_present:
            failures.append("plain story exposed a DSML protocol marker")
        if generation < minimum_output_tokens:
            failures.append(
                f"plain story generated {generation} tokens; "
                f"need at least {minimum_output_tokens}"
            )
        if words < minimum_words:
            failures.append(
                f"plain story contains {words} words; need at least {minimum_words}"
            )

    return {
        "index": measured_index,
        "warmup": warmup,
        "functional": {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "finish_stop": finish_stop,
            "tool_call_count": len(tool_calls),
            "protocol_marker_present": protocol_marker_present,
            "post_response": checkpoint,
        },
        "request_wall_ms": request_wall_ms,
        "context_occupied_tokens": occupied,
        "response": response,
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "response_word_count": words,
        "token_usage": result.token_usage.to_dict(),
        "phase_profiles": phases,
        "story_phase": {
            **story_phase,
            "output_tps": generation * 1000.0 / wall_ms if wall_ms > 0.0 else 0.0,
            "attribution": phase_attribution(story_phase),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workroot", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3.8-27B-UD-Q4_K_S")
    parser.add_argument(
        "--scenario", choices=tuple(SCENARIO_PROMPTS), default="plain-story"
    )
    parser.add_argument("--experiment-id")
    parser.add_argument("--max-output-tokens", type=int, default=1536)
    parser.add_argument("--minimum-context", type=int, default=10_000)
    parser.add_argument("--minimum-output-tokens", type=int, default=400)
    parser.add_argument("--minimum-words", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--prompt")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite profile: {args.output}")
    if not args.server_log.exists():
        raise SystemExit(f"server log does not exist: {args.server_log}")
    if args.max_output_tokens <= 0:
        raise SystemExit("max-output-tokens must be positive")
    if args.warmup < 0 or args.repetitions <= 0:
        raise SystemExit("warmup must be non-negative and repetitions positive")

    prompt = args.prompt or SCENARIO_PROMPTS[args.scenario]
    experiment_id = args.experiment_id or (
        f"agent-dsml-{args.scenario}-{int(time.time())}"
    )
    client = Client(
        provider=ExplicitThinkingAgenticProvider(
            model_name=args.model,
            api_key="ds4-agent-profile",
            base_url=args.base_url,
            timeout=900.0,
            max_retries=0,
        )
    )
    workroot = args.workroot.resolve()
    warmups: list[dict[str, object]] = []
    measured: list[dict[str, object]] = []
    for index in range(args.warmup):
        warmups.append(run_once(
            client=client,
            workroot=workroot,
            server_log=args.server_log,
            scenario=args.scenario,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            minimum_context=args.minimum_context,
            minimum_output_tokens=args.minimum_output_tokens,
            minimum_words=args.minimum_words,
            measured_index=index,
            warmup=True,
        ))
    for index in range(args.repetitions):
        measured.append(run_once(
            client=client,
            workroot=workroot,
            server_log=args.server_log,
            scenario=args.scenario,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            minimum_context=args.minimum_context,
            minimum_output_tokens=args.minimum_output_tokens,
            minimum_words=args.minimum_words,
            measured_index=index,
            warmup=False,
        ))

    functional_pass = all(
        run["functional"]["status"] == "PASS"  # type: ignore[index]
        for run in warmups + measured
    )
    output_tps = [
        float(run["story_phase"]["output_tps"])  # type: ignore[index]
        for run in measured
    ]
    mask_ms = [
        float(run["story_phase"].get("sampling_mask_build_ms", 0.0))  # type: ignore[union-attr]
        for run in measured
    ]
    request_wall = [float(run["request_wall_ms"]) for run in measured]
    record = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "created_at": perf_harness.utc_now(),
        "status": "measured" if functional_pass else "measured_failed_output",
        "hypothesis": (
            "optional DSML SEARCH masking should not dominate a long plain "
            "agent response with tools available and no tool call"
        ),
        "scenario": args.scenario,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "agent": {
            "graph": "bootstrap-wiki",
            "provider": "agentic_openai",
            "model": args.model,
            "prompt": prompt,
            "tool_choice": "auto",
            "reasoning_effort": "none",
            "workroot": str(workroot),
            "max_output_tokens": args.max_output_tokens,
        },
        "requirements": {
            "minimum_context_tokens": args.minimum_context,
            "minimum_output_tokens": args.minimum_output_tokens,
            "minimum_words": args.minimum_words,
        },
        "functional": {"status": "PASS" if functional_pass else "FAIL"},
        "metrics": {
            "output_tps": perf_harness.summary(output_tps),
            "sampling_mask_build_ms": perf_harness.summary(mask_ms),
            "request_wall_ms": perf_harness.summary(request_wall),
        },
        "warmup_runs": warmups,
        "runs": measured,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "status": record["status"],
        "scenario": args.scenario,
        "output_tps": record["metrics"]["output_tps"],
        "word_counts": [run["response_word_count"] for run in measured],
        "generation_tokens": [
            run["story_phase"]["generation_tokens"] for run in measured  # type: ignore[index]
        ],
    }, ensure_ascii=False))
    return 0 if functional_pass else 3


if __name__ == "__main__":
    sys.exit(main())
