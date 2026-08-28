"""Live Agent Wiki coverage for DS4's three SSD checkpoint lifetimes.

Run once against an empty cache (``--phase cold``), restart ds4-server, then
run against the same cache (``--phase warm``).  The companion PowerShell runner
performs that sequence.  These tests intentionally use the packaged DeepAgent
runtime and its real ``bootstrap-wiki`` graph rather than synthesizing HTTP
payloads or short system messages.

For the durable system anchor the runner covers the compatibility matrix:
target-only -> target-only and MTP -> MTP must load, while either mode change
must rebuild.  HDS and regular-skill checkpoints are request-local, so their
live cases verify save/restore in the active server mode rather than reuse
across a server restart.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unittest
import urllib.request
from pathlib import Path

try:
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
except ModuleNotFoundError as exc:
    if exc.name == "agent_wiki":
        raise unittest.SkipTest(
            "agent_wiki is available only in the dedicated Agent SSD environment"
        ) from exc
    raise


MIN_SYSTEM_TOKENS = 8_000
SYSTEM_RESPONSE = "SYSTEM_AGENT_WIKI_OK"
HDS_RESPONSE = "HDS_AGENT_WIKI_OK"
SKILL_CANARY = "AGENT_WIKI_SKILL_SSD_OK"


def runtime_overrides() -> RuntimeConfigOverrides:
    return RuntimeConfigOverrides(
        model=ModelRequestConfigOverride(
            temperature=0.0,
            max_output_tokens=256,
            thinking=False,
        ),
        loop=LoopConfigOverride(max_steps=10),
    )


def make_agent(*, client: Client, workroot: Path) -> DeepAgent:
    """Build the same packaged graph used by /agent's bootstrap frontend."""
    graph = HDSGraphConfig(
        system_config=SystemConfig.load_default(),
        main=HDSNodeConfig(
            name="bootstrap-wiki",
            client=client,
            workroot=workroot,
            runtime=runtime_overrides(),
            children=[WikiGraphNodeConfig()],
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


def run_scenario(
    log_path: Path, name: str, action, *, log_settled=None
) -> tuple[str, str]:
    offset = log_path.stat().st_size if log_path.exists() else 0
    response = action()
    segment = read_log_from(log_path, offset)
    if log_settled is not None:
        deadline = time.monotonic() + 30.0
        while not log_settled(segment) and time.monotonic() < deadline:
            time.sleep(0.1)
            segment = (log_path.read_bytes() if log_path.exists() else b"")[
                offset:
            ].decode("utf-8", errors="replace")
        if not log_settled(segment):
            raise AssertionError(f"{name}: post-response checkpoint did not settle")
    if not segment:
        raise AssertionError(f"{name}: server emitted no log lines")
    return response, segment


def seed_short_agent_anchor(base_url: str, log_path: Path, model_id: str) -> None:
    """Create a deliberately short system anchor before the long agent prompt."""
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "Reply SEED_OK."},
            ],
            "max_tokens": 16,
            "temperature": 0.0,
            "think": False,
        }
    ).encode("utf-8")
    offset = log_path.stat().st_size if log_path.exists() else 0
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300.0) as response:
        response.read()
    segment = read_log_from(log_path, offset)
    boundary = re.search(r"agent-system boundary system_tokens=(\d+)", segment)
    if boundary is None:
        raise AssertionError("failed to identify the short agent-system anchor")
    anchor_tokens = int(boundary.group(1))
    if anchor_tokens <= 0 or anchor_tokens >= MIN_SYSTEM_TOKENS:
        raise AssertionError(
            f"short agent-system anchor has unexpected size: {anchor_tokens}"
        )
    if not re.search(
        rf"kv cache stored tokens={anchor_tokens} .*reason=agent-system",
        segment,
    ):
        raise AssertionError("failed to store the short agent-system anchor")


def assert_system_cache(
    segment: str, phase: str, name: str, warm_cache: str
) -> None:
    boundary_token_counts = [
        int(value)
        for value in re.findall(
            r"agent-system boundary system_tokens=(\d+)", segment
        )
    ]
    cache_token_counts = [
        int(value)
        for value in re.findall(
            r"kv cache hit text(?: RESPPROTO)? tokens=(\d+)", segment
        )
    ]
    system_token_counts = boundary_token_counts + cache_token_counts
    if not system_token_counts:
        raise AssertionError(f"{name}: server did not identify or load a system prompt")
    if max(system_token_counts) < MIN_SYSTEM_TOKENS:
        raise AssertionError(
            f"{name}: expected /agent's long system prompt (at least "
            f"{MIN_SYSTEM_TOKENS} tokens), got {system_token_counts}"
        )
    if phase == "cold":
        if not re.search(r"kv cache stored .*reason=agent-system", segment):
            raise AssertionError(f"{name}: cold run did not store a system prompt")
    elif warm_cache == "refresh":
        stored = re.search(r"kv cache stored .*reason=agent-system", segment)
        if stored is None:
            raise AssertionError(
                f"{name}: MTP mode change did not refresh the system prompt"
            )
        first_hit = segment.find("kv cache hit text")
        if first_hit >= 0 and first_hit < stored.start():
            raise AssertionError(
                f"{name}: MTP mode change loaded an incompatible system prompt"
            )
    else:
        if "kv cache hit text" not in segment:
            raise AssertionError(f"{name}: warm run did not load a system prompt")
        if re.search(r"kv cache stored .*reason=agent-system", segment):
            raise AssertionError(f"{name}: warm run unexpectedly re-prefilled the system prompt")


def assert_resident_post_response_checkpoint(segment: str, name: str) -> None:
    captured = segment.find("resident system frontier captured")
    if captured < 0:
        raise AssertionError(f"{name}: long system frontier was not captured in RAM")
    if "response frontier captured" not in segment[captured:] or not re.search(
        r"response frontier captured .*location=vram", segment[captured:]
    ):
        raise AssertionError(
            f"{name}: pre-thinking Gated DeltaNet frontier was not kept in VRAM"
        )

    finishes = [
        match.start()
        for match in re.finditer(
            r" finish=(?:stop|tool_calls)", segment[captured:]
        )
    ]
    cleanups = [
        match.start()
        for match in re.finditer(
            r"post-response checkpoint rebuilt source=memory-request",
            segment[captured:],
        )
    ]
    if not finishes or len(cleanups) < len(finishes) or any(
        cleanup < finish for finish, cleanup in zip(finishes, cleanups)
    ):
        raise AssertionError(
            f"{name}: checkpoint cleanup did not finish after publishing each response"
        )

    post_response = segment[captured + finishes[0] :]
    if "kv cache hit text" in post_response or "source=disk" in post_response:
        raise AssertionError(
            f"{name}: post-response checkpoint reloaded the system prompt from SSD"
        )
    request_rebuilds = re.findall(
        r"source=memory-request cached=(\d+) response_prefill=(\d+) "
        r"target=(\d+) location=vram",
        post_response,
    )
    if len(request_rebuilds) < len(finishes):
        raise AssertionError(
            f"{name}: post-response checkpoint did not restore the request frontier"
        )
    for cached, response_prefill, target in request_rebuilds:
        cached_n, response_n, target_n = map(int, (cached, response_prefill, target))
        if response_n <= 0 or cached_n + response_n != target_n:
            raise AssertionError(
                f"{name}: expected only the visible response prefill, got "
                f"cached={cached_n} response={response_n} target={target_n}"
            )


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


def run_system_cases(
    client: Client,
    base_url: str,
    model_id: str,
    workroot: Path,
    log_path: Path,
    phase: str,
    warm_cache: str,
) -> list[dict[str, str]]:
    if phase == "cold":
        seed_short_agent_anchor(base_url, log_path, model_id)
    agent = make_agent(client=client, workroot=workroot)
    response, segment = run_scenario(
        log_path,
        "bootstrap-wiki-system",
        lambda: agent.run(
            user_message=(
                "This is a deterministic live test. Follow the configured root "
                "exit procedure now and return exactly "
                f"{SYSTEM_RESPONSE}. Do not inspect or modify files."
            )
        ).response,
        log_settled=lambda log: (
            log.count(" finish=tool_calls") + log.count(" finish=stop") >= 2
            and log.count(
                "post-response checkpoint rebuilt source=memory-request"
            )
            >= log.count(" finish=tool_calls") + log.count(" finish=stop")
        ),
    )
    assert_response(response, SYSTEM_RESPONSE)
    assert_system_cache(segment, phase, "bootstrap-wiki-system", warm_cache)
    assert_resident_post_response_checkpoint(segment, "bootstrap-wiki-system")
    return [{"case": "bootstrap-wiki-system", "response": response}]


def run_hds_cases(
    client: Client,
    workroot: Path,
    log_path: Path,
    phase: str,
    warm_cache: str,
) -> list[dict[str, str]]:
    agent = make_agent(client=client, workroot=workroot)
    response, segment = run_scenario(
        log_path,
        "agent-wiki-hds",
        lambda: agent.run(
            user_message=(
                "Delegate once to the direct child HDS agent-wiki. Tell the child "
                "this is a read-only checkpoint test: it must immediately use its "
                "configured exit procedure and return HDS_CHILD_DONE without using "
                "file tools. HDS_CHILD_DONE is only an intermediate child result and "
                "must never be used or paraphrased as the root's final response. "
                "After the child returns, the root must use its own configured exit "
                f"procedure and return exactly {HDS_RESPONSE}; do not summarize the "
                "checkpoint or return any other text."
            )
        ).response,
    )
    assert_response(response, HDS_RESPONSE)
    if "HDS_CHECKPOINT" not in segment:
        raise AssertionError("agent-wiki-hds: parent history was not saved")
    assert_short_return(segment, "HDS_RETURN", "agent-wiki-hds")
    assert_system_cache(segment, phase, "agent-wiki-hds", warm_cache)
    return [{"case": "agent-wiki-hds", "response": response}]


def run_skill_cases(
    client: Client,
    workroot: Path,
    log_path: Path,
    phase: str,
    warm_cache: str,
) -> list[dict[str, str]]:
    agent = make_agent(client=client, workroot=workroot)
    response, segment = run_scenario(
        log_path,
        "how-read-file-skill",
        lambda: agent.run(
            user_message=(
                "Use the configured how-read-file deep skill to read ssd-canary.txt. "
                "Do not use any write capability. After the skill returns, use the "
                "root exit procedure and include the exact file contents."
            )
        ).response,
    )
    assert_response(response, SKILL_CANARY)
    if "SKILL_CHECKPOINT" not in segment:
        raise AssertionError("how-read-file-skill: skill frontier was not saved")
    assert_short_return(segment, "SKILL_RETURN", "how-read-file-skill")
    assert_system_cache(segment, phase, "how-read-file-skill", warm_cache)
    return [{"case": "how-read-file-skill", "response": response}]


def main() -> None:
    # PowerShell can inherit a legacy CP1252 console although the API and JSON
    # report are UTF-8. Responses may legitimately contain full-width DSML
    # marker characters, so report serialization must not depend on that code
    # page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--model-id", default="Qwen3.8-27B-UD-Q4_K_S")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--workroot", type=Path, required=True)
    parser.add_argument(
        "--warm-cache", choices=("hit", "refresh"), default="hit"
    )
    parser.add_argument(
        "--group", choices=("all", "system", "hds", "skill"),
        default="all",
    )
    args = parser.parse_args()

    workroot = args.workroot.resolve()
    if not workroot.is_dir():
        raise ValueError(f"isolated Agent Wiki workroot does not exist: {workroot}")
    canary = workroot / "ssd-canary.txt"
    if canary.read_text(encoding="utf-8").strip() != SKILL_CANARY:
        raise ValueError(f"unexpected skill canary contents: {canary}")

    client = Client.agentic_openai(
        model=args.model_id,
        api_key="ds4-live-test",
        base_url=args.base_url,
        timeout=900.0,
        max_retries=1,
    )
    report: dict[str, object] = {"phase": args.phase}
    if args.group in ("all", "system"):
        report["system"] = run_system_cases(
            client,
            args.base_url,
            args.model_id,
            workroot,
            args.server_log,
            args.phase,
            args.warm_cache,
        )
    if args.group in ("all", "hds"):
        report["hds"] = run_hds_cases(
            client, workroot, args.server_log, args.phase, args.warm_cache
        )
    if args.group in ("all", "skill"):
        report["skill"] = run_skill_cases(
            client, workroot, args.server_log, args.phase, args.warm_cache
        )
    print("AGENT_SSD_LIVE_REPORT " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
