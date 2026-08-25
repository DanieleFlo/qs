#!/usr/bin/env python3
"""Profile one real /agent story turn and attribute its DSML free-text cost."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from agent_wiki.backend.core.agent.deep_agent import DeepAgent
from agent_wiki.backend.core.client import Client
from agent_wiki.backend.core.config.hds_config import HDSGraphConfig, HDSNodeConfig
from agent_wiki.configs.runtime import (
    LoopConfigOverride,
    ModelRequestConfigOverride,
    RuntimeConfigOverrides,
)
from agent_wiki.configs.settings import SystemConfig
from agent_wiki.wiki import WikiGraphNodeConfig

import perf_harness


DEFAULT_PROMPT = (
    "Genera una storia in italiano di esattamente 300 parole. "
    "Usa la procedura exit-with-info configurata: attiva l'exit skill e inserisci "
    "l'intera storia nel campo message di exit-with-info-tool, usando la forma "
    "{{null}} richiesta dal tool. Non usare altri skill o tool, non leggere o "
    "modificare file e non aggiungere commenti alla storia."
)


def make_agent(*, client: Client, workroot: Path,
               max_output_tokens: int) -> DeepAgent:
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


def read_new_log(path: Path, offset: int) -> str:
    deadline = time.monotonic() + 5.0
    text = ""
    while time.monotonic() < deadline:
        text = path.read_bytes()[offset:].decode("utf-8", errors="replace")
        if "ds4-server: phase profile " in text:
            time.sleep(0.2)
            return path.read_bytes()[offset:].decode("utf-8", errors="replace")
        time.sleep(0.1)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workroot", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3.6-27B-Q4_K_S")
    parser.add_argument(
        "--experiment-id",
        default="constraint-m5-agent-dsml-story-300-current-001",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--minimum-context", type=int, default=10_000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite profile: {args.output}")
    if not args.server_log.exists():
        raise SystemExit(f"server log does not exist: {args.server_log}")

    client = Client.agentic_openai(
        model=args.model,
        api_key="ds4-agent-profile",
        base_url=args.base_url,
        timeout=900.0,
        max_retries=0,
    )
    if args.max_output_tokens <= 0:
        raise SystemExit("max-output-tokens must be positive")
    agent = make_agent(
        client=client, workroot=args.workroot.resolve(),
        max_output_tokens=args.max_output_tokens,
    )
    offset = args.server_log.stat().st_size
    started = time.monotonic()
    result = agent.run(user_message=args.prompt)
    request_wall_ms = (time.monotonic() - started) * 1000.0
    segment = read_new_log(args.server_log, offset)
    phases = perf_harness.parse_server_phase_profiles(segment)
    constrained_phases = [
        phase for phase in phases
        if int(phase.get("filter_calls", 0)) > 0
    ]
    if not constrained_phases:
        raise SystemExit("agent turn emitted no constrained phase profile")
    story_phase = max(
        constrained_phases,
        key=lambda phase: int(phase["generation_tokens"]),
    )
    occupied = context_tokens(str(story_phase["context_span"]))
    generation = int(story_phase["generation_tokens"])
    wall_ms = float(story_phase["wall_ms"])
    response = result.response
    generation_error = re.search(
        r"finish=error[^\r\n]*error=\"([^\"]+)\"", segment
    )
    functional_status = "FAIL" if generation_error else "PASS"
    status = (
        "invalid_context" if occupied < args.minimum_context else
        "measured_failed_output" if generation_error else
        "measured"
    )
    record = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "created_at": perf_harness.utc_now(),
        "status": status,
        "hypothesis": (
            "after static-mask reduction, the residual DSML free-text cost in the "
            "real /agent path is no longer dominated by vocabulary masking"
        ),
        "attempts": 1,
        "agent": {
            "graph": "bootstrap-wiki",
            "provider": "agentic_openai",
            "model": args.model,
            "prompt": args.prompt,
            "workroot": str(args.workroot.resolve()),
            "max_output_tokens": args.max_output_tokens,
        },
        "request_wall_ms": request_wall_ms,
        "minimum_context_tokens": args.minimum_context,
        "context_occupied_tokens": occupied,
        "functional": {
            "status": functional_status,
            "error": generation_error.group(1) if generation_error else None,
        },
        "response": response,
        "response_word_count": word_count(response),
        "token_usage": result.token_usage.to_dict(),
        "phase_profiles": phases,
        "story_phase": {
            **story_phase,
            "output_tps": generation * 1000.0 / wall_ms if wall_ms > 0.0 else 0.0,
            "attribution": phase_attribution(story_phase),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "status": record["status"],
        "context_occupied_tokens": occupied,
        "response_word_count": record["response_word_count"],
        "story_generation_tokens": generation,
        "story_output_tps": record["story_phase"]["output_tps"],
    }, ensure_ascii=False))
    if occupied < args.minimum_context:
        return 2
    return 3 if generation_error else 0


if __name__ == "__main__":
    sys.exit(main())
