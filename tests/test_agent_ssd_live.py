"""Live Agent Wiki coverage for DS4's three SSD checkpoint lifetimes.

Run once against an empty cache (``--phase cold``), restart ds4-server, then
run against the same cache (``--phase warm``).  The companion PowerShell runner
performs that sequence.  These tests intentionally use the packaged DeepAgent
runtime rather than synthesizing HTTP payloads.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

from agent_wiki.backend.core.agent.deep_agent import DeepAgent
from agent_wiki.backend.core.client import Client
from agent_wiki.backend.core.config.hds_config import HDSGraphConfig, HDSNodeConfig
from agent_wiki.configs.runtime import (
    CheckpointConfigOverride,
    LoopConfigOverride,
    ModelRequestConfigOverride,
    RuntimeConfigOverrides,
)
from agent_wiki.configs.settings import SystemConfig


HERE = Path(__file__).resolve().parent
WORKROOT = HERE / "agent_ssd_live"


def runtime_overrides() -> RuntimeConfigOverrides:
    return RuntimeConfigOverrides(
        model=ModelRequestConfigOverride(
            temperature=0.0,
            max_output_tokens=256,
            thinking=False,
        ),
        loop=LoopConfigOverride(max_steps=10),
        checkpoint=CheckpointConfigOverride(enabled=False),
    )


def system_config() -> SystemConfig:
    defaults = SystemConfig()
    return SystemConfig(
        config=replace(defaults.config, inject_capability_summaries=False)
    )


def make_agent(
    *,
    client: Client,
    root: str,
    children: tuple[str, ...] = (),
    root_tool_choice: str = "auto",
    child_tool_choice: str = "auto",
) -> DeepAgent:
    child_nodes = [
        HDSNodeConfig(
            name=name,
            client=client,
            workroot=WORKROOT,
            runtime=runtime_overrides(),
            preserve_history=False,
            tool_choice=child_tool_choice,
        )
        for name in children
    ]
    graph = HDSGraphConfig(
        system_config=system_config(),
        main=HDSNodeConfig(
            name=root,
            client=client,
            workroot=WORKROOT,
            runtime=runtime_overrides(),
            children=child_nodes,
            tool_choice=root_tool_choice,
        ),
    )
    return DeepAgent(node_config=graph)


def assert_response(actual: str, expected: str) -> None:
    if expected not in actual:
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def read_log_from(path: Path, offset: int) -> str:
    deadline = time.monotonic() + 5.0
    text = ""
    while time.monotonic() < deadline:
        raw = path.read_bytes() if path.exists() else b""
        text = raw[offset:].decode("utf-8", errors="replace")
        if "prompt done" in text or "SKILL_RETURN" in text or "HDS_RETURN" in text:
            time.sleep(0.1)
            raw = path.read_bytes()
            return raw[offset:].decode("utf-8", errors="replace")
        time.sleep(0.1)
    return text


def run_scenario(log_path: Path, name: str, action) -> tuple[str, str]:
    offset = log_path.stat().st_size if log_path.exists() else 0
    response = action()
    segment = read_log_from(log_path, offset)
    if not segment:
        raise AssertionError(f"{name}: server emitted no log lines")
    return response, segment


def assert_system_cache(segment: str, phase: str, name: str) -> None:
    if phase == "cold":
        if not re.search(r"kv cache stored .*reason=agent-system", segment):
            raise AssertionError(f"{name}: cold run did not store a system prompt")
    else:
        if "kv cache hit text" not in segment:
            raise AssertionError(f"{name}: warm run did not load a system prompt")
        if re.search(r"kv cache stored .*reason=agent-system", segment):
            raise AssertionError(f"{name}: warm run unexpectedly re-prefilled the system prompt")


def assert_short_return(segment: str, marker: str, name: str) -> None:
    match = re.search(
        rf"{marker} .*restored_tokens=(\d+).*result_prefill_tokens=(\d+)",
        segment,
    )
    if match is None:
        raise AssertionError(f"{name}: missing {marker} metrics")
    restored, result_prefill = map(int, match.groups())
    if result_prefill <= 0 or result_prefill >= restored:
        raise AssertionError(
            f"{name}: expected only the short result prefill, got "
            f"restored={restored} result={result_prefill}"
        )


def run_system_cases(client: Client, log_path: Path, phase: str) -> list[dict[str, str]]:
    reports: list[dict[str, str]] = []
    for suffix in ("a", "b"):
        root = f"ssd-root-{suffix}"
        expected = f"SYSTEM_{suffix.upper()}_OK"
        agent = make_agent(client=client, root=root)
        response, segment = run_scenario(
            log_path,
            root,
            lambda agent=agent: agent.run(user_message="Answer now.").response,
        )
        assert_response(response, expected)
        assert_system_cache(segment, phase, root)
        reports.append({"case": root, "response": response})
    return reports


def run_hds_cases(client: Client, log_path: Path, phase: str) -> list[dict[str, str]]:
    reports: list[dict[str, str]] = []
    agent = make_agent(
        client=client,
        root="ssd-hds-root",
        children=("ssd-child-a", "ssd-child-b"),
    )
    assert_response(
        agent.run(user_message="Initialize history only.").response,
        "READY_OK",
    )
    for suffix in ("a", "b"):
        expected = f"HDS_CHILD_{suffix.upper()}_OK"
        response, segment = run_scenario(
            log_path,
            f"hds-{suffix}",
            lambda agent=agent, suffix=suffix: agent.run(
                user_message=f"Run CHILD_{suffix.upper()} now."
            ).response,
        )
        assert_response(response, expected)
        if "HDS_CHECKPOINT" not in segment:
            raise AssertionError(f"hds-{suffix}: parent history was not saved")
        assert_short_return(segment, "HDS_RETURN", f"hds-{suffix}")
        assert_system_cache(segment, phase, f"hds-{suffix}-child-system")
        reports.append({"case": f"hds-{suffix}", "response": response})
    return reports


def run_skill_cases(
    client: Client, log_path: Path, phase: str
) -> list[dict[str, str]]:
    reports: list[dict[str, str]] = []
    for suffix in ("a", "b"):
        expected = f"SKILL_PARENT_{suffix.upper()}_OK"
        agent = make_agent(client=client, root=f"ssd-skill-root-{suffix}")
        response, segment = run_scenario(
            log_path,
            f"skill-{suffix}",
            lambda agent=agent, suffix=suffix: agent.run(
                user_message=f"Run SKILL_{suffix.upper()} now."
            ).response,
        )
        assert_response(response, expected)
        if "SKILL_CHECKPOINT" not in segment:
            raise AssertionError(f"skill-{suffix}: skill frontier was not saved")
        assert_short_return(segment, "SKILL_RETURN", f"skill-{suffix}")
        assert_system_cache(segment, phase, f"skill-{suffix}-system")
        reports.append({"case": f"skill-{suffix}", "response": response})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument(
        "--group", choices=("all", "system", "hds", "skill"), default="all"
    )
    args = parser.parse_args()

    client = Client.agentic_openai(
        model="deepseek-v4-flash",
        api_key="ds4-live-test",
        base_url=args.base_url,
        timeout=180.0,
        max_retries=1,
    )
    report: dict[str, object] = {"phase": args.phase}
    if args.group in ("all", "system"):
        report["system"] = run_system_cases(client, args.server_log, args.phase)
    if args.group in ("all", "hds"):
        report["hds"] = run_hds_cases(client, args.server_log, args.phase)
    if args.group in ("all", "skill"):
        report["skill"] = run_skill_cases(client, args.server_log, args.phase)
    print("AGENT_SSD_LIVE_REPORT " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
